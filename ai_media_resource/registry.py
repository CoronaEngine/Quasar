"""
媒体资源注册表（MediaResourceRegistry）

统一管理所有媒体资源（图片、视频、音频）的 file_id -> URL 映射。

核心设计：
- file_id 是内部流转的唯一标识
- 仅在返回客户端 JSON 或调用上游 API 时解析为真实 URL
- 与 session TTL 对齐，支持自动过期清理
- 异步任务执行委托给 TaskExecutor

**超时管理**: 支持使用 DeadlineContext 的剩余时间。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from ..ai_config.ai_config import get_ai_config
from .adapter_factory import (
    resolve_cache_url,
)
from .record import MediaRecord
from .result import StorageResult
from .task_executor import (
    TaskStatus,
    get_task_executor,
)

logger = logging.getLogger(__name__)

_MEDIA_TASK_SCHEDULER: Any = None
_MEDIA_TASK_JOBS: Dict[str, str] = {}
_MEDIA_TASK_LOCK = threading.Lock()


def set_media_task_scheduler(scheduler: Any) -> None:
    """Install an optional GenerationScheduler for media generation tasks."""
    global _MEDIA_TASK_SCHEDULER
    with _MEDIA_TASK_LOCK:
        _MEDIA_TASK_SCHEDULER = scheduler
        if scheduler is None:
            _MEDIA_TASK_JOBS.clear()


def get_media_task_scheduler() -> Any:
    with _MEDIA_TASK_LOCK:
        return _MEDIA_TASK_SCHEDULER


def _remember_media_task_job(file_id: str, job_id: str) -> None:
    with _MEDIA_TASK_LOCK:
        _MEDIA_TASK_JOBS[file_id] = job_id


def _forget_media_task_job(file_id: str) -> None:
    with _MEDIA_TASK_LOCK:
        _MEDIA_TASK_JOBS.pop(file_id, None)


def _get_media_task_job(file_id: str) -> tuple[Any, str]:
    with _MEDIA_TASK_LOCK:
        return _MEDIA_TASK_SCHEDULER, _MEDIA_TASK_JOBS.get(file_id, "")


def _submit_media_task_job(
    *,
    scheduler: Any,
    file_id: str,
    task_fn: Callable[[], Union[str, StorageResult]],
    resource_type: str,
    session_id: str,
) -> bool:
    submit = getattr(scheduler, "submit", None)
    if not callable(submit):
        return False

    def _run_media_task(job: Any) -> Dict[str, Any]:
        result = task_fn()
        url = getattr(result, "url", result)
        expire_time = getattr(result, "url_expire_time", None)
        return {
            "media_file_id": file_id,
            "resource_type": resource_type,
            "content_url": url,
            "url_expire_time": expire_time,
        }

    payload = {
        "job_id": f"media-{file_id}",
        "job_type": "media_resource_task",
        "session_id": session_id,
        "batch_id": file_id,
        "resource_type": resource_type,
        "_runtime_context": {
            "stage_order": ("submit",),
            "stage_handlers": {"submit": _run_media_task},
        },
    }
    submitted = submit(payload)
    job_id = ""
    if isinstance(submitted, dict):
        job_id = str(submitted.get("job_id") or "")
        if submitted.get("status") == "waiting_user" and submitted.get("error"):
            logger.warning("[%s] media scheduler rejected task: %s", file_id, submitted.get("error"))
            return False
    else:
        job_id = str(getattr(submitted, "job_id", "") or "")
    if not job_id:
        return False
    _remember_media_task_job(file_id, job_id)
    return True


class MediaResourceRegistry:
    """
    媒体资源注册表（单例）

    统一管理 file_id -> MediaRecord 映射，支持：
    - submit(): 提交异步任务，立即返回 file_id
    - register(): 注册媒体资源，立即返回 file_id
    - resolve(): 获取 URL（异步任务会阻塞等待）
    - get_by_file_id(): 获取完整的媒体记录
    - get_session_media(): 获取会话的所有媒体记录
    - cleanup_session(): 清理会话资源
    """

    _instance: Optional["MediaResourceRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MediaResourceRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 从配置获取参数
        session_ttl = 86400
        max_items_per_session = 50
        try:
            cfg = get_ai_config()
            session_cfg = getattr(cfg, "session", None)
            if session_cfg and hasattr(session_cfg, "ttl_seconds"):
                session_ttl = session_cfg.ttl_seconds
        except Exception:
            pass

        # 核心数据结构
        self._records: Dict[str, MediaRecord] = {}
        self._session_file_ids: Dict[str, List[str]] = {}
        self._session_timestamps: Dict[str, int] = {}

        self._records_lock = threading.Lock()
        self._session_ttl = session_ttl
        self._max_items_per_session = max_items_per_session
        self._cleanup_interval = 300
        self._last_cleanup = int(time.time())
        self._initialized = True

        logger.info(
            f"MediaResourceRegistry 初始化完成，session_ttl: {session_ttl}s"
        )

    def _generate_file_id(self) -> str:
        """生成 12 位十六进制 file_id"""
        return uuid.uuid4().hex[:12]

    def _touch_session(self, session_id: str) -> None:
        """更新会话最后访问时间（必须在锁内调用）"""
        self._session_timestamps[session_id] = int(time.time())

    def _add_to_session(self, session_id: str, file_id: str) -> None:
        """将 file_id 添加到会话列表（必须在锁内调用）"""
        if session_id not in self._session_file_ids:
            self._session_file_ids[session_id] = []
        self._session_file_ids[session_id].append(file_id)
        self._touch_session(session_id)

        # 限制每个会话的媒体数量
        if len(self._session_file_ids[session_id]) > self._max_items_per_session:
            old_file_id = self._session_file_ids[session_id].pop(0)
            self._records.pop(old_file_id, None)

    def _cleanup_if_needed(self) -> None:
        """定期清理过期会话（必须在锁内调用）"""
        current_time = int(time.time())
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        expired_sessions = [
            sid
            for sid, ts in self._session_timestamps.items()
            if current_time - ts > self._session_ttl
        ]

        for sid in expired_sessions:
            file_ids = self._session_file_ids.pop(sid, [])
            for fid in file_ids:
                self._records.pop(fid, None)
            self._session_timestamps.pop(sid, None)

        if expired_sessions:
            logger.debug(f"清理了 {len(expired_sessions)} 个过期会话的媒体记录")

        self._last_cleanup = current_time

    # ==================== 异步任务提交 ====================

    def submit(
        self,
        task_fn: Callable[[], Union[str, StorageResult]],
        resource_type: str,
        session_id: str,
        content_text: str = "",
        parameter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        提交异步任务，立即返回 file_id

        参数:
        - task_fn: 无参 callable，返回 URL（str）或 StorageResult
        - resource_type: 资源类型（image/video/audio）
        - session_id: 会话 ID
        - content_text: 媒体描述文本
        - parameter: 附加参数

        返回:
        - file_id: 12 位十六进制字符串
        """
        file_id = self._generate_file_id()

        def wrapped_task():
            result = task_fn()
            # 处理返回值
            # 兼容不同路径导入造成的 StorageResult 类型不一致 (Duck Typing)
            if hasattr(result, "url") and hasattr(result, "url_expire_time"):
                url = result.url
                url_expire_time = result.url_expire_time
            elif isinstance(result, StorageResult):
                url = result.url
                url_expire_time = result.url_expire_time
            else:
                url = result
                url_expire_time = None
            # 更新记录
            with self._records_lock:
                if file_id in self._records:
                    self._records[file_id].content_url = url
                    self._records[file_id].url_expire_time = url_expire_time
            return result

        # 创建记录
        record = MediaRecord(
            file_id=file_id,
            session_id=session_id,
            resource_type=resource_type,
            content_text=content_text,
            source="tool",
            timestamp=int(time.time()),
            parameter=parameter or {},
            task_id=file_id,
        )

        with self._records_lock:
            self._cleanup_if_needed()
            self._records[file_id] = record
            self._add_to_session(session_id, file_id)

        scheduler = get_media_task_scheduler()
        scheduled = False
        scheduler_error = ""
        if scheduler is not None:
            try:
                scheduled = _submit_media_task_job(
                    scheduler=scheduler,
                    file_id=file_id,
                    task_fn=wrapped_task,
                    resource_type=resource_type,
                    session_id=session_id,
                )
            except Exception as exc:  # noqa: BLE001
                scheduler_error = str(exc)
                logger.warning("[%s] media scheduler submit failed: %s", file_id, exc)

        if scheduler is not None and not scheduled:
            if not scheduler_error:
                scheduler_error = "media scheduler rejected task"
            with self._records_lock:
                if file_id in self._records:
                    self._records[file_id].error = scheduler_error
            logger.warning("[%s] media task not started because scheduler rejected it: %s", file_id, scheduler_error)
        elif not scheduled:
            executor = get_task_executor()
            executor.submit(
                task_id=file_id,
                task_fn=wrapped_task,
                metadata={"resource_type": resource_type, "session_id": session_id},
            )

        logger.debug(f"[{file_id}] 任务已提交: {resource_type}, session={session_id}")
        return file_id

    # ==================== 媒体注册 ====================

    def register(
        self,
        session_id: str,
        content_url: str,
        resource_type: str,
        content_text: str = "",
        parameter: Optional[Dict[str, Any]] = None,
        url_expire_time: Optional[int] = None,
    ) -> str:
        """
        注册媒体资源，立即返回 file_id

        参数:
        - session_id: 会话 ID
        - content_url: 媒体 URL
        - resource_type: 资源类型（image/video/audio）
        - content_text: 媒体描述
        - parameter: 附加参数
        - url_expire_time: URL 过期时间（秒级时间戳）

        返回:
        - file_id: 12 位十六进制字符串
        """
        file_id = self._generate_file_id()

        record = MediaRecord(
            file_id=file_id,
            session_id=session_id,
            resource_type=resource_type,
            content_url=content_url,
            url_expire_time=url_expire_time,
            content_text=content_text,
            source="upload",
            timestamp=int(time.time()),
            parameter=parameter or {},
        )

        with self._records_lock:
            self._cleanup_if_needed()
            self._records[file_id] = record
            self._add_to_session(session_id, file_id)

        logger.debug(f"[{file_id}] 注册: {resource_type}, session={session_id}")
        return file_id

    def register_batch(
        self,
        session_id: str,
        parts: List[Dict[str, Any]],
    ) -> List[str]:
        """
        批量注册媒体资源

        参数:
        - session_id: 会话 ID
        - parts: 媒体 part 列表，每个 part 需包含 content_type, content_url

        返回:
        - file_id 列表
        """
        file_ids = []
        for part in parts:
            content_type = part.get("content_type")
            content_url = part.get("content_url")

            if content_type not in ("image", "video", "audio"):
                continue
            if not content_url:
                continue

            file_id = self.register(
                session_id=session_id,
                content_url=content_url,
                resource_type=content_type,
                content_text=part.get("content_text", ""),
                parameter=part.get("parameter"),
            )
            file_ids.append(file_id)

        return file_ids

    # ==================== URL 解析 ====================

    def resolve(
        self,
        file_id: str,
        timeout: float | None = None,
        encode_to_base64: bool | None = None,
        return_original_url: bool = False,
    ) -> str:
        """
        获取 file_id 对应的 URL，异步任务会阻塞等待完成

        **超时管理**: 优先使用 DeadlineContext 的剩余时间（如果存在）。

        参数:
        - file_id: 资源 ID
        - timeout: 等待超时时间（秒），None 表示由 TaskExecutor 智能推断
        - encode_to_base64: 保留参数，暂未使用（base64 转换在上层处理）
        - return_original_url: 是否返回原始云端 URL（而非 base64）
          - False (默认): 返回 base64 data URI（安全，不泄露上游 URL）
          - True: 返回原始云端 URL（用于需要 HTTP URL 的上游 API）

        返回:
        - url: 最终 URL

        异常:
        - KeyError: file_id 不存在
        - TimeoutError: 任务超时（DeadlineExceeded 或等待超时）
        - RuntimeError: 任务执行失败
        """
        with self._records_lock:
            if file_id not in self._records:
                raise KeyError(f"未知的 file_id: {file_id}")
            record = self._records[file_id]
            self._touch_session(record.session_id)

            # 已有 URL，直接返回
            if record.content_url:
                url = record.content_url
                if url.startswith("cache://"):
                    resolved_url = resolve_cache_url(url, return_original_url=return_original_url)
                    if resolved_url is None:
                        raise RuntimeError(f"缓存已过期: {url}")
                    return resolved_url
                return url
            if record.error:
                raise RuntimeError(f"任务 {file_id} 失败: {record.error}")

        # 有关联的异步任务，等待完成
        if record.task_id:
            scheduler, scheduler_job_id = _get_media_task_job(record.task_id)
            if scheduler is not None and scheduler_job_id:
                try:
                    wait = getattr(scheduler, "wait", None)
                    status_fn = getattr(scheduler, "status", None)
                    status = (
                        wait(scheduler_job_id, timeout=timeout or 300.0)
                        if callable(wait)
                        else status_fn(scheduler_job_id)
                        if callable(status_fn)
                        else {}
                    )
                    if isinstance(status, dict):
                        state = str(status.get("status") or "")
                        if state == "done":
                            with self._records_lock:
                                current = self._records.get(file_id)
                                url = current.content_url if current else ""
                            if not url:
                                result = status.get("result") or {}
                                url = str(result.get("content_url") or "")
                            if url.startswith("cache://"):
                                resolved_url = resolve_cache_url(url, return_original_url=return_original_url)
                                if resolved_url is None:
                                    raise RuntimeError(f"缓存已过期: {url}")
                                return resolved_url
                            if url:
                                return url
                            raise RuntimeError(f"任务 {scheduler_job_id} 已完成但未返回 URL")
                        if state in {"failed", "cancelled", "abandoned"}:
                            error = str(status.get("error") or state)
                            with self._records_lock:
                                if file_id in self._records:
                                    self._records[file_id].error = error
                            raise RuntimeError(f"任务 {scheduler_job_id} 失败: {error}")
                    raise TimeoutError(f"任务 {scheduler_job_id} 未在限定时间内完成")
                except Exception as e:
                    with self._records_lock:
                        if file_id in self._records:
                            self._records[file_id].error = str(e)
                    raise

            executor = get_task_executor()
            try:
                # timeout 为 None 时，由 executor.wait() 根据任务类型智能推断
                result = executor.wait(record.task_id, timeout=timeout)
                # 从结果中提取 URL
                # 兼容不同路径导入造成的 StorageResult 类型不一致 (Duck Typing)
                if hasattr(result, "url") and hasattr(result, "url_expire_time"):
                    url = result.url
                elif isinstance(result, StorageResult):
                    url = result.url
                else:
                    url = result
                # 处理 cache:// URL
                if url.startswith("cache://"):
                    resolved_url = resolve_cache_url(url, return_original_url=return_original_url)
                    if resolved_url is None:
                        raise RuntimeError(f"缓存已过期: {url}")
                    return resolved_url
                return url
            except Exception as e:
                # 更新错误信息
                with self._records_lock:
                    if file_id in self._records:
                        self._records[file_id].error = str(e)
                raise

        raise RuntimeError(f"file_id {file_id} 没有有效的 URL")

    def resolve_with_expire_time(
        self,
        file_id: str,
        encode_to_base64: bool | None = None,
    ) -> Dict[str, Any]:
        """
        获取 file_id 对应的 URL 和过期时间

        **超时管理**: 使用 DeadlineContext 的剩余时间（如果存在），
        否则使用默认超时。

        参数:
        - file_id: 资源 ID
        - encode_to_base64: 保留参数，暂未使用（base64 转换在上层处理）

        返回:
        - {"url": "...", "url_expire_time": 秒级时间戳或 None}
        """
        url = self.resolve(file_id, encode_to_base64=encode_to_base64)

        with self._records_lock:
            record = self._records.get(file_id)
            url_expire_time = record.url_expire_time if record else None

        return {"url": url, "url_expire_time": url_expire_time}

    # ==================== 记录查询 ====================

    def get_by_file_id(self, file_id: str) -> Optional[MediaRecord]:
        """获取媒体记录"""
        with self._records_lock:
            record = self._records.get(file_id)
            if record:
                self._touch_session(record.session_id)
            return record

    def get_status(self, file_id: str) -> TaskStatus:
        """查询任务状态"""
        with self._records_lock:
            if file_id not in self._records:
                raise KeyError(f"未知的 file_id: {file_id}")
            record = self._records[file_id]

            # 已有 URL，已完成
            if record.content_url:
                return TaskStatus.DONE
            # 有错误，失败
            if record.error:
                return TaskStatus.ERROR
            # 有任务 ID，查询任务状态
            if record.task_id:
                scheduler, scheduler_job_id = _get_media_task_job(record.task_id)
                if scheduler is not None and scheduler_job_id:
                    status_fn = getattr(scheduler, "status", None)
                    if callable(status_fn):
                        status = status_fn(scheduler_job_id)
                        state = str(status.get("status") or "") if isinstance(status, dict) else ""
                        if state == "done":
                            return TaskStatus.DONE
                        if state in {"failed", "cancelled", "abandoned", "not_found"}:
                            return TaskStatus.ERROR
                        return TaskStatus.PENDING
                executor = get_task_executor()
                return executor.get_status(record.task_id)

            return TaskStatus.ERROR

    def get_session_media(
        self,
        session_id: str,
        limit: Optional[int] = None,
        source: Optional[str] = None,
    ) -> List[MediaRecord]:
        """
        获取会话的媒体记录列表

        参数:
        - session_id: 会话 ID
        - limit: 返回数量限制
        - source: 来源过滤（"tool" / "upload"）

        返回:
        - MediaRecord 列表（按时间顺序）
        """
        with self._records_lock:
            self._touch_session(session_id)
            file_ids = self._session_file_ids.get(session_id, [])

            records = []
            for fid in file_ids:
                record = self._records.get(fid)
                if record is None:
                    continue
                if source is not None and record.source != source:
                    continue
                records.append(record)

            if limit is not None and limit < len(records):
                records = records[-limit:]

            return records

    def get_session_parts(
        self,
        session_id: str,
        limit: Optional[int] = None,
        source: Optional[str] = None,
        resolved_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """获取会话的媒体 part 列表"""
        records = self.get_session_media(session_id, limit=limit, source=source)
        parts = []
        for record in records:
            if resolved_only and not record.content_url:
                continue
            parts.append(record.to_part())
        return parts

    # ==================== 会话管理 ====================

    def cleanup(self, file_id: str) -> None:
        """清理单个媒体记录"""
        with self._records_lock:
            record = self._records.pop(file_id, None)
            if record:
                session_file_ids = self._session_file_ids.get(record.session_id, [])
                if file_id in session_file_ids:
                    session_file_ids.remove(file_id)
                # 清理关联的任务
                if record.task_id:
                    _forget_media_task_job(record.task_id)
                    executor = get_task_executor()
                    executor.cleanup(record.task_id)

    def cleanup_session(self, session_id: str) -> int:
        """清理指定会话的所有媒体记录"""
        with self._records_lock:
            file_ids = self._session_file_ids.pop(session_id, [])
            self._session_timestamps.pop(session_id, None)

            count = 0
            executor = get_task_executor()
            for fid in file_ids:
                record = self._records.pop(fid, None)
                if record is not None:
                    count += 1
                    if record.task_id:
                        _forget_media_task_job(record.task_id)
                        executor.cleanup(record.task_id)

            return count

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """获取会话的统计信息"""
        with self._records_lock:
            file_ids = self._session_file_ids.get(session_id, [])

            pending = 0
            completed = 0
            error = 0
            media_counts: Dict[str, int] = {}
            source_counts: Dict[str, int] = {}

            for fid in file_ids:
                record = self._records.get(fid)
                if record is None:
                    continue

                # 统计状态
                if record.content_url:
                    completed += 1
                elif record.error:
                    error += 1
                elif record.task_id:
                    try:
                        scheduler, scheduler_job_id = _get_media_task_job(record.task_id)
                        if scheduler is not None and scheduler_job_id:
                            status_fn = getattr(scheduler, "status", None)
                            scheduler_status = (
                                status_fn(scheduler_job_id)
                                if callable(status_fn)
                                else {"status": "pending"}
                            )
                            state = (
                                str(scheduler_status.get("status") or "")
                                if isinstance(scheduler_status, dict)
                                else ""
                            )
                            status = (
                                TaskStatus.DONE
                                if state == "done"
                                else TaskStatus.ERROR
                                if state in {"failed", "cancelled", "abandoned", "not_found"}
                                else TaskStatus.PENDING
                            )
                        else:
                            executor = get_task_executor()
                            status = executor.get_status(record.task_id)
                        if status == TaskStatus.PENDING:
                            pending += 1
                        elif status == TaskStatus.DONE:
                            completed += 1
                        else:
                            error += 1
                    except KeyError:
                        error += 1

                media_counts[record.resource_type] = (
                    media_counts.get(record.resource_type, 0) + 1
                )
                source_counts[record.source] = source_counts.get(record.source, 0) + 1

            return {
                "pending_tasks": pending,
                "completed_tasks": completed,
                "error_tasks": error,
                "media_counts": media_counts,
                "source_counts": source_counts,
                "total": len(file_ids),
            }


# ==============================================================================
# 模块级便捷函数
# ==============================================================================

_registry: Optional[MediaResourceRegistry] = None
_registry_lock = threading.Lock()


def get_media_registry() -> MediaResourceRegistry:
    """获取 MediaResourceRegistry 单例"""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = MediaResourceRegistry()
    return _registry


def reset_media_registry() -> None:
    """重置 MediaResourceRegistry（用于测试）"""
    global _registry
    with _registry_lock:
        _registry = None


__all__ = [
    "MediaResourceRegistry",
    "get_media_registry",
    "get_media_task_scheduler",
    "reset_media_registry",
    "set_media_task_scheduler",
]
