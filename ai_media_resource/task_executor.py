"""
异步任务执行器

管理异步任务的提交、执行和结果获取。
从 MediaResourceRegistry 中剥离，专注于线程池和 Future 管理。

**超时管理**: 支持 DeadlineContext，优先使用 deadline 剩余时间。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from ..ai_config.ai_config import get_ai_config
from .result import StorageResult

logger = logging.getLogger(__name__)


# 默认等待超时（秒）- 与 DeadlineConfig.DEFAULT 保持一致
_DEFAULT_WAIT_TIMEOUT = 300.0


def _infer_timeout_from_metadata(metadata: Dict[str, Any]) -> float:
    """
    根据任务元数据推断合理的超时时间

    从 metadata 中的 resource_type 推断，与 DeadlineConfig 保持一致。
    """
    resource_type = metadata.get("resource_type")
    if not resource_type:
        return _DEFAULT_WAIT_TIMEOUT

    try:
        from app.utils.deadline import DeadlineConfig

        return DeadlineConfig.get(resource_type)
    except ImportError:
        # 降级：手动映射
        mapping = {
            "image": 300.0,  # 5分钟
            "video": 600.0,  # 10分钟
            "music": 300.0,  # 5分钟
            "audio": 300.0,
            "speech": 120.0,  # 2分钟
            "text": 120.0,
            "detection": 120.0,
        }
        return mapping.get(resource_type.lower(), _DEFAULT_WAIT_TIMEOUT)


def _get_deadline_remaining(default: float, reserve: float = 5.0) -> float:
    """从 DeadlineContext 获取剩余时间"""
    # ... (移除默认值 150.0，改为必须传入)
    try:
        from app.utils.deadline import (
            get_remaining,
            is_in_task_context,
            warn_nested_call,
        )

        if is_in_task_context():
            warn_nested_call("TaskExecutor.wait()")
        return get_remaining(reserve=reserve, default=default)
    except ImportError:
        return default
    except Exception:
        return default


class TaskStatus(Enum):
    """任务状态"""

    PENDING = "pending"
    DONE = "done"
    ERROR = "error"


@dataclass
class TaskRecord:
    """异步任务记录"""

    task_id: str
    future: Future
    submit_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> TaskStatus:
        """获取当前状态"""
        if not self.future.done():
            return TaskStatus.PENDING
        return TaskStatus.ERROR if self.error else TaskStatus.DONE

    @property
    def is_done(self) -> bool:
        return self.future.done()


class TaskExecutor:
    """
    异步任务执行器（单例）

    管理线程池，提供：
    - submit(): 提交任务到线程池
    - wait(): 等待任务完成
    - get_status(): 查询任务状态
    - get_metrics(): 获取任务时序指标
    """

    _instance: Optional["TaskExecutor"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TaskExecutor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        max_workers = self._get_max_workers()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="task_executor_"
        )
        self._tasks: Dict[str, TaskRecord] = {}
        self._tasks_lock = threading.Lock()
        self._initialized = True

        logger.info(f"TaskExecutor 初始化完成，线程池大小: {max_workers}")

    def _get_max_workers(self) -> int:
        """获取线程池大小"""
        env_val = os.environ.get("TASK_EXECUTOR_WORKERS")
        if env_val:
            try:
                iv = int(env_val)
                if iv > 0:
                    return iv
            except ValueError:
                pass

        try:
            cfg = get_ai_config()
            session_cfg = getattr(cfg, "session", None)
            if session_cfg and hasattr(session_cfg, "file_registry_max_workers"):
                val = getattr(session_cfg, "file_registry_max_workers", None)
                if isinstance(val, int) and val > 0:
                    return val
        except Exception:
            pass

        cpu = os.cpu_count() or 4
        return min(max(4, cpu), 32)

    def submit(
        self,
        task_id: str,
        task_fn: Callable[[], Union[str, StorageResult, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        提交任务到线程池

        参数:
        - task_id: 任务 ID
        - task_fn: 无参 callable
        - metadata: 任务元数据

        返回:
        - task_id
        """

        def wrapped_task():
            with self._tasks_lock:
                if task_id in self._tasks:
                    self._tasks[task_id].start_time = time.time()
            try:
                result = task_fn()
                with self._tasks_lock:
                    if task_id in self._tasks:
                        self._tasks[task_id].result = result
                logger.debug(f"[{task_id}] 任务完成")
                return result
            except Exception as e:
                logger.error(f"[{task_id}] 任务失败: {e}", exc_info=True)
                with self._tasks_lock:
                    if task_id in self._tasks:
                        self._tasks[task_id].error = str(e)
                raise
            finally:
                with self._tasks_lock:
                    if task_id in self._tasks:
                        self._tasks[task_id].end_time = time.time()

        submit_time = time.time()
        future = self._executor.submit(wrapped_task)

        record = TaskRecord(
            task_id=task_id,
            future=future,
            submit_time=submit_time,
            metadata=metadata or {},
        )

        with self._tasks_lock:
            self._tasks[task_id] = record

        logger.debug(f"[{task_id}] 任务已提交")
        return task_id

    def wait(self, task_id: str, timeout: float | None = None) -> Any:
        """
        等待任务完成并返回结果

        **超时策略**:
        1. 优先使用 DeadlineContext 的剩余时间
        2. 如果指定了 timeout，使用该值
        3. 否则根据 metadata.resource_type 智能推断
        4. 最后降级到 _DEFAULT_WAIT_TIMEOUT
        """
        with self._tasks_lock:
            if task_id not in self._tasks:
                raise KeyError(f"未知的 task_id: {task_id}")
            record = self._tasks[task_id]

        # 智能推断超时
        if timeout is None:
            timeout = _infer_timeout_from_metadata(record.metadata)

        # 尝试从 DeadlineContext 获取
        actual_timeout = _get_deadline_remaining(default=timeout, reserve=5.0)

        try:
            result = record.future.result(timeout=actual_timeout)
            return result
        except FutureTimeoutError:
            resource_type = record.metadata.get("resource_type", "unknown")
            raise TimeoutError(
                f"任务 {task_id} ({resource_type}) 超时（{actual_timeout:.1f}秒）"
            ) from None
        except Exception as e:
            error_msg = record.error or str(e)
            raise RuntimeError(f"任务 {task_id} 失败: {error_msg}") from e

    def get_status(self, task_id: str) -> TaskStatus:
        """非阻塞查询任务状态"""
        with self._tasks_lock:
            if task_id not in self._tasks:
                raise KeyError(f"未知的 task_id: {task_id}")
            return self._tasks[task_id].status

    def get_result(self, task_id: str) -> Optional[Any]:
        """获取已完成任务的结果（不阻塞）"""
        with self._tasks_lock:
            if task_id not in self._tasks:
                raise KeyError(f"未知的 task_id: {task_id}")
            record = self._tasks[task_id]
            if record.is_done and record.error is None:
                return record.result
            return None

    def get_metrics(self, task_id: str) -> Dict[str, Optional[float | str]]:
        """
        获取任务时序指标

        返回:
        - status: 当前状态
        - wait_time: 等待时间 (start_time - submit_time)
        - exec_time: 执行时间 (end_time - start_time)
        - total_time: 总耗时 (end_time - submit_time)
        - error: 错误信息
        """
        with self._tasks_lock:
            if task_id not in self._tasks:
                raise KeyError(f"未知的 task_id: {task_id}")
            record = self._tasks[task_id]

            wait_time = None
            exec_time = None
            total_time = None

            if record.submit_time and record.start_time:
                wait_time = record.start_time - record.submit_time
            if record.start_time and record.end_time:
                exec_time = record.end_time - record.start_time
            if record.submit_time and record.end_time:
                total_time = record.end_time - record.submit_time

            return {
                "status": record.status.value,
                "wait_time": wait_time,
                "exec_time": exec_time,
                "total_time": total_time,
                "error": record.error,
            }

    def cleanup(self, task_id: str) -> None:
        """清理单个任务记录"""
        with self._tasks_lock:
            self._tasks.pop(task_id, None)

    def cleanup_completed(self) -> int:
        """清理所有已完成的任务"""
        with self._tasks_lock:
            completed = [tid for tid, rec in self._tasks.items() if rec.future.done()]
            for tid in completed:
                del self._tasks[tid]
            return len(completed)

    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池"""
        self._executor.shutdown(wait=wait)
        logger.info("TaskExecutor 已关闭")


# ==============================================================================
# 模块级便捷函数
# ==============================================================================

_executor: Optional[TaskExecutor] = None
_executor_lock = threading.Lock()


def get_task_executor() -> TaskExecutor:
    """获取 TaskExecutor 单例"""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = TaskExecutor()
    return _executor


def reset_task_executor() -> None:
    """重置 TaskExecutor（用于测试）"""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None


__all__ = [
    "TaskStatus",
    "TaskRecord",
    "TaskExecutor",
    "get_task_executor",
    "reset_task_executor",
]
