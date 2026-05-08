"""
会话缓存管理器

提供会话进度追踪的读写接口，采用单例模式。
支持内存 + Redis + MongoDB 三层存储架构。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional

from .models import (
    StepRetryInfo,
    StepInfo,
    AccountUsageRecord,
    DeadlineInfo,
    SessionCache,
)


# 数据库存储层（懒加载以避免循环导入）
def _get_redis_session_store():
    from database import redis_session_store

    return redis_session_store


def _get_mongo_session_store():
    from database import mongo_session_store

    return mongo_session_store


logger = logging.getLogger(__name__)


def _now_timestamp() -> float:
    """获取当前时间的Unix时间戳"""
    return time.time()


def _now_iso() -> str:
    """获取当前时间的ISO格式字符串（用于步骤日志）"""
    from datetime import datetime

    return datetime.now().isoformat()


class SessionCacheManager:
    """
    会话缓存管理器（单例）

    提供线程安全的会话状态管理：
    - 写入接口：供工作流调用
    - 读取接口：供 API 调用
    - 数据库同步：自动同步到 Redis/MongoDB
    """

    def __init__(self, enable_db_sync: bool = True) -> None:
        self._store: Dict[str, SessionCache] = {}
        self._lock = threading.RLock()
        self._enable_db_sync = enable_db_sync
        # 创建用于数据库操作的后台线程池（避免阻塞主流程）
        self._db_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="session_db_"
        )

    def _sync_state_to_redis(self, cache: SessionCache) -> None:
        """异步同步会话状态到 Redis（非阻塞）"""
        if not self._enable_db_sync:
            return

        # 复制必要数据避免线程安全问题
        cache_dict = cache.to_dict()
        session_id = cache.session_id
        state = cache.state
        created_at_iso = cache.created_at_iso
        updated_at_iso = cache.updated_at_iso
        input_type = cache.input_type
        error_message = cache.error_message
        total_cost = cache.total_cost
        total_calls = len(cache.account_usages)

        def _sync():
            try:
                store = _get_redis_session_store()
                # 1. 保存基础状态（使用 ISO 格式）
                store.save_session_state(
                    session_id=session_id,
                    state=state,
                    created_at_iso=created_at_iso,
                    updated_at_iso=updated_at_iso,
                    input_type=input_type,
                    error_message=error_message,
                    total_cost=total_cost,
                    total_calls=total_calls,
                )
                # 2. 保存完整数据（包含所有字段）
                store.save_complete_data(session_id, cache_dict)
            except Exception as e:
                logger.warning(f"同步会话状态到 Redis 失败: {e}")

        # 提交到线程池异步执行
        self._db_executor.submit(_sync)

    def _sync_progress_to_redis(self, session_id: str, cache: SessionCache) -> None:
        """异步同步会话进度到 Redis（非阻塞）"""
        if not self._enable_db_sync:
            return

        # 复制必要数据避免线程安全问题
        progress = cache.progress
        current_step = progress.current_step
        total_steps = progress.total_steps
        step_name = progress.step_name
        step_message = progress.step_message
        progress_percent = progress.progress_percent
        is_retrying = progress.is_retrying
        current_attempt = progress.current_attempt
        max_attempts = progress.max_attempts
        total_retries = progress.total_retries
        estimated_remaining_seconds = progress.estimated_remaining_seconds
        steps_history = [step.to_dict() for step in progress.steps_history]

        def _sync():
            try:
                store = _get_redis_session_store()
                # 1. 保存进度基础信息
                store.save_session_progress(
                    session_id=session_id,
                    current_step=current_step,
                    total_steps=total_steps,
                    step_name=step_name,
                    step_message=step_message,
                    progress_percent=progress_percent,
                    is_retrying=is_retrying,
                    current_attempt=current_attempt,
                    max_attempts=max_attempts,
                    total_retries=total_retries,
                    estimated_remaining_seconds=estimated_remaining_seconds,
                )
                # 2. 更新步骤历史
                store.update_steps_history(session_id, steps_history)
            except Exception as e:
                logger.warning(f"同步会话进度到 Redis 失败: {e}")

        # 提交到线程池异步执行
        self._db_executor.submit(_sync)

    def _save_snapshot_to_mongo(self, cache: SessionCache) -> None:
        """异步保存会话快照到 MongoDB（非阻塞）"""
        if not self._enable_db_sync:
            return

        # 复制数据避免线程安全问题
        cache_dict = cache.to_dict()
        session_id = cache.session_id

        def _save():
            try:
                store = _get_mongo_session_store()
                store.save_snapshot(cache_dict)
                logger.info(f"会话 {session_id} 快照已保存到 MongoDB")
            except Exception as e:
                logger.warning(f"保存会话快照到 MongoDB 失败: {e}")

        # 提交到线程池异步执行
        self._db_executor.submit(_save)

    # ========================================================================
    # 写入接口（由工作流调用）
    # ========================================================================

    def init_session(
        self,
        session_id: str,
        input_type: str,
        parameters: Dict[str, Any],
        workflow_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """工作流启动时初始化会话

        Args:
            session_id: 会话ID
            input_type: 输入类型 (chat/workflow/single)
            parameters: 输入参数字典
            workflow_state: 工作流初始状态（可选，包含 function_id, prompt, images 等字段）
        """
        with self._lock:
            now = _now_timestamp()
            cache = SessionCache(
                session_id=session_id,
                state="idle",
                created_at=now,
                updated_at=now,
                input_type=input_type,
                input_parameters=parameters.copy(),
            )

            # 如果提供了工作流状态，提取关键字段
            if workflow_state:
                cache.function_id = workflow_state.get("function_id")
                cache.prompt = workflow_state.get("prompt", "")
                cache.images = workflow_state.get("images", [])
                cache.additional_type = workflow_state.get("additional_type")
                cache.bounding_box = workflow_state.get("bounding_box")
                cache.resolution = workflow_state.get("resolution", "1:1")
                cache.image_size = workflow_state.get("image_size", "2K")
                cache.metadata = workflow_state.get("metadata", {})

            self._store[session_id] = cache
            # 同步到 Redis
            self._sync_state_to_redis(cache)
            # 同时保存到 MongoDB，确保所有会话从一开始就有记录
            self._save_snapshot_to_mongo(cache)

    def update_state(self, session_id: str, state: str) -> None:
        """工作流更新状态

        Args:
            session_id: 会话ID
            state: 状态 (idle/running/completed/failed/cancelled)
        """
        with self._lock:
            if session_id not in self._store:
                return
            cache = self._store[session_id]
            cache.state = state
            cache.updated_at = _now_timestamp()

            # 同步到 Redis
            self._sync_state_to_redis(cache)

            # 如果会话完成或失败，保存快照到 MongoDB
            if state in ("completed", "failed", "cancelled"):
                self._save_snapshot_to_mongo(cache)

    def update_progress(
        self,
        session_id: str,
        current_step: int,
        total_steps: int,
        step_name: str,
        message: str,
        progress_percent: Optional[float] = None,
    ) -> None:
        """工作流更新进度

        Args:
            session_id: 会话ID
            current_step: 当前步骤编号（从1开始）
            total_steps: 总步骤数
            step_name: 步骤名称
            message: 进度消息
            progress_percent: 进度百分比（0-100），如果不提供则自动计算
        """
        with self._lock:
            if session_id not in self._store:
                return
            cache = self._store[session_id]
            progress = cache.progress
            progress.current_step = current_step
            progress.total_steps = total_steps
            progress.step_name = step_name
            progress.step_message = message

            if progress_percent is not None:
                progress.progress_percent = progress_percent
            elif total_steps > 0:
                progress.progress_percent = (current_step / total_steps) * 100.0

            cache.updated_at = _now_timestamp()

            # 同步到 Redis
            self._sync_progress_to_redis(session_id, cache)

    def record_step_start(
        self,
        session_id: str,
        step_name: str,
        step_number: int,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> None:
        """工作流记录步骤开始

        Args:
            session_id: 会话ID
            step_name: 步骤名称
            step_number: 步骤编号
            attempt: 当前尝试次数（从1开始）
            max_attempts: 最大尝试次数
        """
        with self._lock:
            if session_id not in self._store:
                return
            cache = self._store[session_id]
            progress = cache.progress

            # 查找是否已有该步骤
            existing_step = None
            for step in progress.steps_history:
                if step.name == step_name and step.step == step_number:
                    existing_step = step
                    break

            if existing_step:
                # 更新现有步骤
                if attempt > 1:
                    existing_step.status = "retrying"
                    if not existing_step.retry_info:
                        existing_step.retry_info = StepRetryInfo(
                            attempt_count=attempt,
                            max_attempts=max_attempts,
                        )
                    else:
                        existing_step.retry_info.attempt_count = attempt
                        existing_step.retry_info.max_attempts = max_attempts
                else:
                    existing_step.status = "running"
                    existing_step.started_at = _now_iso()
            else:
                # 创建新步骤
                step_info = StepInfo(
                    step=step_number,
                    name=step_name,
                    status="running",
                    started_at=_now_iso(),
                )
                if attempt > 1:
                    step_info.status = "retrying"
                    step_info.retry_info = StepRetryInfo(
                        attempt_count=attempt,
                        max_attempts=max_attempts,
                    )
                progress.steps_history.append(step_info)

            # 更新全局进度状态
            progress.is_retrying = attempt > 1
            progress.current_attempt = attempt
            progress.max_attempts = max_attempts

            cache.updated_at = _now_timestamp()

    def record_step_retry(
        self,
        session_id: str,
        step_name: str,
        step_number: int,
        error: str,
        next_attempt: int,
    ) -> None:
        """工作流记录重试

        Args:
            session_id: 会话ID
            step_name: 步骤名称
            step_number: 步骤编号
            error: 错误信息
            next_attempt: 下次尝试次数
        """
        with self._lock:
            if session_id not in self._store:
                return
            cache = self._store[session_id]
            progress = cache.progress

            # 查找步骤
            for step in progress.steps_history:
                if step.name == step_name and step.step == step_number:
                    if not step.retry_info:
                        step.retry_info = StepRetryInfo(
                            attempt_count=next_attempt - 1,
                            max_attempts=3,
                        )

                    step.retry_info.last_error = error
                    step.retry_info.retry_history.append(
                        {
                            "attempt": next_attempt - 1,
                            "failed_at": _now_iso(),
                            "error": error,
                        }
                    )
                    step.status = "retrying"

                    # 更新全局重试计数
                    progress.total_retries += 1
                    break

            cache.updated_at = _now_timestamp()

    def record_step_complete(
        self,
        session_id: str,
        step_name: str,
        step_number: int,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """工作流记录步骤完成

        Args:
            session_id: 会话ID
            step_name: 步骤名称
            step_number: 步骤编号
            success: 是否成功
            error: 错误信息（失败时）
        """
        with self._lock:
            if session_id not in self._store:
                return
            cache = self._store[session_id]
            progress = cache.progress

            # 查找步骤
            for step in progress.steps_history:
                if step.name == step_name and step.step == step_number:
                    step.status = "completed" if success else "failed"
                    step.completed_at = _now_iso()

                    # 计算耗时
                    if step.started_at:
                        try:
                            start = datetime.fromisoformat(step.started_at)
                            end = datetime.fromisoformat(step.completed_at)
                            step.duration_ms = int((end - start).total_seconds() * 1000)
                        except Exception:
                            pass

                    # 如果失败，记录错误
                    if not success and error:
                        if not step.retry_info:
                            step.retry_info = StepRetryInfo(
                                attempt_count=1,
                                max_attempts=1,
                            )
                        step.retry_info.last_error = error

                    break

            # 重置重试状态
            progress.is_retrying = False
            progress.current_attempt = 1

            cache.updated_at = _now_timestamp()

    def append_output(
        self, session_id: str, output_type: str, content: Dict[str, Any]
    ) -> None:
        """工作流添加输出结果

        Args:
            session_id: 会话ID
            output_type: 输出类型 (text/image/video/audio等)
            content: 输出内容字典
        """
        with self._lock:
            if session_id not in self._store:
                return
            cache = self._store[session_id]

            # 使用统一的时间字段命名
            now_ts = time.time()
            from datetime import datetime

            now_iso = datetime.fromtimestamp(now_ts).isoformat()

            cache.outputs.append(
                {
                    "type": output_type,
                    "content": content,
                    "created_at": now_ts,
                    "created_at_iso": now_iso,
                }
            )
            cache.updated_at = _now_timestamp()

            # 异步同步到 Redis（非阻塞）
            if self._enable_db_sync:
                output_data = cache.outputs[-1].copy()

                def _sync():
                    try:
                        store = _get_redis_session_store()
                        store.append_output(session_id, output_data)
                    except Exception as e:
                        logger.warning(f"同步输出到 Redis 失败: {e}")

                self._db_executor.submit(_sync)

    def set_error(self, session_id: str, error: str) -> None:
        """工作流记录错误

        Args:
            session_id: 会话ID
            error: 错误信息
        """
        with self._lock:
            if session_id not in self._store:
                return
            cache = self._store[session_id]
            cache.error_message = error
            cache.updated_at = _now_timestamp()

    def record_account_usage(
        self,
        session_id: str,
        account_id: str,
        account_name: str,
        model: Optional[str],
        price: float,
        latency_ms: float,
        success: bool,
    ) -> None:
        """记录账户使用到会话缓存

        Args:
            session_id: 会话ID
            account_id: 账户ID
            account_name: 账户名称（adapter_type）
            model: 模型名称
            price: 单次调用价格
            latency_ms: 延迟（毫秒）
            success: 是否成功
        """
        with self._lock:
            if session_id not in self._store:
                logger.warning(f"会话 {session_id} 不存在，跳过账户使用记录")
                return

            cache = self._store[session_id]
            record = AccountUsageRecord(
                account_id=account_id,
                account_name=account_name,
                model=model,
                timestamp=time.time(),
                price=price,
                latency_ms=latency_ms,
                success=success,
            )
            cache.account_usages.append(record)
            cache.updated_at = _now_timestamp()

            # 异步同步费用统计和账户使用记录到 Redis（非阻塞）
            if self._enable_db_sync:
                record_dict = record.to_dict()
                cost = price if success else 0

                def _sync():
                    try:
                        store = _get_redis_session_store()
                        # 1. 增量更新费用统计
                        store.increment_cost(session_id, cost, success)
                        # 2. 追加账户使用记录
                        store.append_account_usage(session_id, record_dict)
                    except Exception as e:
                        logger.warning(f"同步账户使用到 Redis 失败: {e}")

                self._db_executor.submit(_sync)

    def record_deadline_info(
        self,
        session_id: str,
        deadline: float,
        start_time: float,
        stage_timings: list,
        success: bool,
        is_timeout: bool,
    ) -> None:
        """记录 Deadline 超时管理信息

        Args:
            session_id: 会话ID
            deadline: 绝对截止时间（时间戳）
            start_time: 开始时间（时间戳）
            stage_timings: 阶段耗时记录列表
            success: 是否成功完成
            is_timeout: 是否超时
        """
        with self._lock:
            if session_id not in self._store:
                logger.debug(f"会话 {session_id} 不存在，跳过 deadline 信息记录")
                return

            cache = self._store[session_id]
            elapsed_ms = (time.time() - start_time) * 1000
            deadline_seconds = deadline - start_time

            cache.deadline_info = DeadlineInfo(
                deadline=deadline,
                start_time=start_time,
                deadline_seconds=deadline_seconds,
                is_timeout=is_timeout,
                elapsed_ms=elapsed_ms,
                stage_timings=stage_timings,
            )
            cache.updated_at = _now_timestamp()

            # 输出阶段耗时摘要日志
            if stage_timings:
                summary = cache.deadline_info.stage_summary
                by_stage = summary.get("by_stage", {})

                # 构建阶段详细信息字符串
                stage_details = []
                for stage_name, duration_ms in by_stage.items():
                    stage_details.append(f"{stage_name}={duration_ms:.0f}ms")

                stage_info = " | ".join(stage_details) if stage_details else ""

                logger.info(
                    f"[Session {session_id}] Deadline 统计: "
                    f"total={elapsed_ms:.0f}ms | stages={summary.get('stage_count', 0)} | "
                    f"timeout={is_timeout}"
                )

                # 如果有阶段信息，输出详细耗时
                if stage_info:
                    logger.info(f"[Session {session_id}] 阶段耗时详情: {stage_info}")

    def get_deadline_info(self, session_id: str) -> Optional[DeadlineInfo]:
        """获取会话的 Deadline 信息

        Args:
            session_id: 会话ID

        Returns:
            DeadlineInfo 对象，如果不存在返回 None
        """
        with self._lock:
            if session_id not in self._store:
                return None
            return self._store[session_id].deadline_info

    # ========================================================================
    # 读取接口（由 API 调用）
    # ========================================================================

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """API查询会话状态

        Args:
            session_id: 会话ID

        Returns:
            包含会话状态信息的字典
        """
        with self._lock:
            if session_id not in self._store:
                return {
                    "code": 404,
                    "msg": "会话不存在",
                    "data": None,
                }

            cache = self._store[session_id]
            progress = cache.progress

            return {
                "code": 0,
                "msg": "ok",
                "data": {
                    "session_id": session_id,
                    "state": cache.state,
                    "created_at": cache.created_at,
                    "updated_at": cache.updated_at,
                    "current_step": progress.step_name,
                    "total_steps": progress.total_steps,
                    "progress_percent": round(progress.progress_percent, 2),
                    "error_message": cache.error_message,
                    # 实时费用统计
                    "total_cost": round(cache.total_cost, 4),
                    "total_calls": len(cache.account_usages),
                    "successful_calls": sum(
                        1 for r in cache.account_usages if r.success
                    ),
                },
            }

    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """API查询进度详情

        Args:
            session_id: 会话ID

        Returns:
            包含详细进度信息的字典
        """
        with self._lock:
            if session_id not in self._store:
                return {
                    "code": 404,
                    "msg": "会话不存在",
                    "data": None,
                }

            cache = self._store[session_id]
            progress_dict = cache.progress.to_dict()
            progress_dict["session_id"] = session_id
            # 添加实时费用统计
            progress_dict["total_cost"] = round(cache.total_cost, 4)
            progress_dict["total_calls"] = len(cache.account_usages)
            progress_dict["successful_calls"] = sum(
                1 for r in cache.account_usages if r.success
            )

            return {
                "code": 0,
                "msg": "ok",
                "data": progress_dict,
            }

    def get_input(self, session_id: str) -> Dict[str, Any]:
        """API查询输入参数

        Args:
            session_id: 会话ID

        Returns:
            包含输入参数的字典
        """
        with self._lock:
            if session_id not in self._store:
                return {
                    "code": 404,
                    "msg": "会话不存在",
                    "data": None,
                }

            cache = self._store[session_id]

            return {
                "code": 0,
                "msg": "ok",
                "data": {
                    "session_id": session_id,
                    "input_type": cache.input_type,
                    "parameters": cache.input_parameters,
                    "submitted_at": cache.created_at,
                },
            }

    def get_output(self, session_id: str) -> Dict[str, Any]:
        """API查询输出结果

        Args:
            session_id: 会话ID

        Returns:
            包含输出结果的字典
        """
        with self._lock:
            if session_id not in self._store:
                return {
                    "code": 404,
                    "msg": "会话不存在",
                    "data": None,
                }

            cache = self._store[session_id]

            # 判断输出状态
            output_status = "pending"
            if cache.state == "completed":
                output_status = "completed"
            elif cache.state == "failed":
                output_status = "failed"
            elif len(cache.outputs) > 0:
                output_status = "partial"

            return {
                "code": 0,
                "msg": "ok",
                "data": {
                    "session_id": session_id,
                    "status": output_status,
                    "outputs": cache.outputs,
                    "total_outputs": len(cache.outputs),
                    "completed_at": (
                        cache.updated_at if cache.state == "completed" else None
                    ),
                },
            }

    def get_snapshot(self, session_id: str) -> Dict[str, Any]:
        """API查询会话完整快照

        Args:
            session_id: 会话ID

        Returns:
            包含所有会话信息的字典
        """
        with self._lock:
            if session_id not in self._store:
                return {
                    "code": 404,
                    "msg": "会话不存在",
                    "data": None,
                }

            cache = self._store[session_id]

            return {
                "code": 0,
                "msg": "ok",
                "data": cache.to_dict(),
            }

    def get_accounts(self, session_id: str) -> Dict[str, Any]:
        """API查询会话账户使用信息

        Args:
            session_id: 会话ID

        Returns:
            包含账户使用详情和统计摘要的字典
        """
        with self._lock:
            if session_id not in self._store:
                return {
                    "code": 404,
                    "msg": "会话不存在",
                    "data": None,
                }

            cache = self._store[session_id]

            return {
                "code": 0,
                "msg": "ok",
                "data": {
                    "session_id": session_id,
                    "account_usages": [r.to_dict() for r in cache.account_usages],
                    "usage_summary": cache.usage_summary,
                },
            }

    # ========================================================================
    # 管理接口
    # ========================================================================

    def exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with self._lock:
            return session_id in self._store

    def clear(self, session_id: str) -> None:
        """清除指定会话"""
        with self._lock:
            self._store.pop(session_id, None)

    def clear_all(self) -> None:
        """清除所有会话（用于测试）"""
        with self._lock:
            self._store.clear()

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self._lock:
            return {
                "total_sessions": len(self._store),
                "running_sessions": sum(
                    1 for c in self._store.values() if c.state == "running"
                ),
                "completed_sessions": sum(
                    1 for c in self._store.values() if c.state == "completed"
                ),
                "failed_sessions": sum(
                    1 for c in self._store.values() if c.state == "failed"
                ),
            }


# 全局单例
_SESSION_CACHE_MANAGER: Optional[SessionCacheManager] = None
_MANAGER_LOCK = threading.Lock()


def get_session_cache_manager() -> SessionCacheManager:
    """获取全局会话缓存管理器实例（懒加载单例）"""
    global _SESSION_CACHE_MANAGER
    if _SESSION_CACHE_MANAGER is None:
        with _MANAGER_LOCK:
            if _SESSION_CACHE_MANAGER is None:
                _SESSION_CACHE_MANAGER = SessionCacheManager()
    return _SESSION_CACHE_MANAGER


__all__ = [
    "SessionCacheManager",
    "get_session_cache_manager",
]
