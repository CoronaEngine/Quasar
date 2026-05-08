"""
统一并发控制模块

提供 Session 级别的并发限流和资源管理：
- 基于 BoundedSemaphore 的 per-session 并发控制
- 自动清理过期的 session 信号量
- 线程安全的信号量管理
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

from ..ai_config.ai_config import get_ai_config, AIConfig

logger = logging.getLogger(__name__)


class SessionConcurrencyManager:
    """
    Session 级别并发管理器

    特性：
    - 基于 BoundedSemaphore 的 per-session 并发控制
    - 自动清理过期的 session 信号量（与会话 TTL 同步）
    - 线程安全
    - 支持配置化的并发限制
    """

    def __init__(
        self,
        max_concurrent_requests: int | None = None,
        session_ttl_seconds: int | None = None,
        cleanup_interval: int = 300,  # 每 5 分钟清理一次
    ) -> None:
        # 从配置获取默认值
        if max_concurrent_requests is None or session_ttl_seconds is None:
            try:
                config = get_ai_config()
                if max_concurrent_requests is None:
                    max_concurrent_requests = config.session.max_concurrent_requests
                if session_ttl_seconds is None:
                    session_ttl_seconds = config.session.ttl_seconds
            except Exception:
                if max_concurrent_requests is None:
                    max_concurrent_requests = 0  # 默认不限流
                if session_ttl_seconds is None:
                    session_ttl_seconds = 86400

        self._lock = threading.RLock()
        self._semaphores: Dict[str, threading.BoundedSemaphore] = {}
        self._timestamps: Dict[str, int] = {}  # 记录最后访问时间（秒）
        self._max_concurrent = (
            max_concurrent_requests
            if max_concurrent_requests and max_concurrent_requests > 0
            else 0
        )
        self._session_ttl = session_ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = int(time.time())

    def _get_semaphore(self, session_id: str) -> threading.BoundedSemaphore | None:
        """获取或创建 session 的信号量（内部方法，需在锁内调用）"""
        if self._max_concurrent <= 0:
            return None

        current_time = int(time.time())
        self._timestamps[session_id] = current_time

        # 定期清理
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time

        sem = self._semaphores.get(session_id)
        if sem is None:
            sem = threading.BoundedSemaphore(self._max_concurrent)
            self._semaphores[session_id] = sem
            logger.debug(
                f"为 session {session_id} 创建信号量，限制: {self._max_concurrent}"
            )

        return sem

    def _cleanup_expired(self) -> None:
        """清理过期的 session 信号量（内部方法，需在锁内调用）"""
        current_time = int(time.time())
        expired = [
            sid
            for sid, ts in self._timestamps.items()
            if current_time - ts > self._session_ttl
        ]
        for sid in expired:
            self._semaphores.pop(sid, None)
            self._timestamps.pop(sid, None)
        if expired:
            logger.debug(f"清理了 {len(expired)} 个过期的 session 信号量")

    def acquire(
        self,
        session_id: str,
        timeout: float | None = None,
    ) -> Tuple[bool, threading.BoundedSemaphore | None]:
        """
        获取 session 的并发许可

        Args:
            session_id: 会话 ID
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            (acquired, semaphore): 是否成功获取，以及信号量对象（用于释放）
            如果不限流，返回 (True, None)
        """
        with self._lock:
            sem = self._get_semaphore(session_id)

        if sem is None:
            return True, None

        # 在锁外等待，避免阻塞其他 session
        if timeout is not None:
            acquired = sem.acquire(timeout=timeout)
        else:
            acquired = sem.acquire(blocking=True)

        if not acquired:
            logger.warning(f"Session {session_id} 并发获取超时")

        return acquired, sem

    def release(self, semaphore: threading.BoundedSemaphore | None) -> None:
        """释放并发许可"""
        if semaphore is not None:
            try:
                semaphore.release()
            except ValueError:
                # 信号量已经被释放，忽略
                pass

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self._lock:
            return {
                "active_sessions": len(self._semaphores),
                "max_concurrent_per_session": self._max_concurrent,
            }

    def clear_session(self, session_id: str) -> None:
        """清理指定 session 的信号量"""
        with self._lock:
            self._semaphores.pop(session_id, None)
            self._timestamps.pop(session_id, None)

    def clear_all(self) -> None:
        """清理所有信号量（用于测试或重置）"""
        with self._lock:
            self._semaphores.clear()
            self._timestamps.clear()


# 全局单例
_CONCURRENCY_MANAGER: Optional[SessionConcurrencyManager] = None
_MANAGER_LOCK = threading.Lock()


def get_concurrency_manager() -> SessionConcurrencyManager:
    """获取全局并发管理器实例（懒加载单例）"""
    global _CONCURRENCY_MANAGER
    if _CONCURRENCY_MANAGER is None:
        with _MANAGER_LOCK:
            if _CONCURRENCY_MANAGER is None:
                _CONCURRENCY_MANAGER = SessionConcurrencyManager()
    return _CONCURRENCY_MANAGER


def get_acquire_timeout(cfg: AIConfig | None = None) -> float | None:
    """从配置获取并发获取超时时间"""
    if cfg is None:
        try:
            cfg = get_ai_config()
        except Exception:
            return None

    try:
        if hasattr(cfg, "polling") and hasattr(cfg.polling, "max_wait_seconds"):
            val = cfg.polling.max_wait_seconds
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
    except Exception:
        pass

    return None


@contextmanager
def session_concurrency(session_id: str, cfg: AIConfig | None = None):
    """
    Session 并发控制上下文管理器

    用法:
        with session_concurrency(session_id, cfg) as acquired:
            if not acquired:
                raise RuntimeError("并发繁忙，请稍后重试")
            # 执行实际操作

    Args:
        session_id: 会话 ID
        cfg: AI 配置（可选，用于获取超时时间）

    Yields:
        bool: 是否成功获取并发许可
    """
    manager = get_concurrency_manager()
    timeout = get_acquire_timeout(cfg)
    acquired, sem = manager.acquire(session_id, timeout=timeout)

    try:
        yield acquired
    finally:
        manager.release(sem)


__all__ = [
    "SessionConcurrencyManager",
    "get_concurrency_manager",
    "get_acquire_timeout",
    "session_concurrency",
]
