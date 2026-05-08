"""
线程安全的会话存储
支持多用户并发访问
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Sequence, Optional
from langchain_core.messages import BaseMessage


class ThreadSafeConversationStore:
    """
    线程安全的会话存储

    特性：
    - 使用 RLock 保护所有读写操作
    - 支持会话过期清理
    - 支持最大会话数限制
    """

    def __init__(
        self,
        max_sessions: int | None = None,
        session_ttl_seconds: int | None = None,
        max_messages_per_session: int | None = None,
    ) -> None:
        # 获取配置默认值
        if (
            max_sessions is None
            or session_ttl_seconds is None
            or max_messages_per_session is None
        ):
            try:
                from ..ai_config.ai_config import (
                    get_ai_config,
                )

                config = get_ai_config()
                if max_sessions is None:
                    max_sessions = config.session.max_sessions
                if session_ttl_seconds is None:
                    session_ttl_seconds = config.session.ttl_seconds
                if max_messages_per_session is None:
                    max_messages_per_session = config.session.max_messages_per_session
            except Exception:
                # 回退到硬编码默认值
                if max_sessions is None:
                    max_sessions = 10000
                if session_ttl_seconds is None:
                    session_ttl_seconds = 86400
                if max_messages_per_session is None:
                    max_messages_per_session = 100

        self._lock = threading.RLock()
        self._sessions: Dict[str, List[BaseMessage]] = {}
        self._timestamps: Dict[str, int] = {}  # 记录最后访问时间（秒）
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl_seconds
        self._max_messages = max_messages_per_session

    def snapshot(self, session_id: str) -> List[BaseMessage]:
        """获取会话历史快照（线程安全）"""
        with self._lock:
            self._timestamps[session_id] = int(time.time())
            return list(self._sessions.get(session_id, []))

    def update(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        """更新会话历史（线程安全）"""
        with self._lock:
            # 限制消息数量，保留最新的
            msg_list = list(messages)
            if len(msg_list) > self._max_messages:
                msg_list = msg_list[-self._max_messages:]

            self._sessions[session_id] = msg_list
            self._timestamps[session_id] = int(time.time())

            # 检查是否需要清理过期会话
            self._cleanup_if_needed()

    def delete(self, session_id: str) -> bool:
        """删除会话（线程安全）"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._timestamps.pop(session_id, None)
                return True
            return False

    def exists(self, session_id: str) -> bool:
        """检查会话是否存在（线程安全）"""
        with self._lock:
            return session_id in self._sessions

    def get_session_count(self) -> int:
        """获取当前会话数量"""
        with self._lock:
            return len(self._sessions)

    def _cleanup_if_needed(self) -> None:
        """清理过期或超量的会话（必须在锁内调用）"""
        current_time = int(time.time())

        # 清理过期会话
        expired = [
            sid
            for sid, ts in self._timestamps.items()
            if current_time - ts > self._session_ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._timestamps.pop(sid, None)

        # 如果仍然超过最大数量，删除最旧的
        if len(self._sessions) > self._max_sessions:
            # 按时间戳排序，删除最旧的
            sorted_sessions = sorted(self._timestamps.items(), key=lambda x: x[1])
            to_remove = len(self._sessions) - self._max_sessions
            for sid, _ in sorted_sessions[:to_remove]:
                self._sessions.pop(sid, None)
                self._timestamps.pop(sid, None)

    def clear_all(self) -> None:
        """清空所有会话（用于测试或重置）"""
        with self._lock:
            self._sessions.clear()
            self._timestamps.clear()


# 全局线程安全存储实例
_THREAD_SAFE_STORE: Optional[ThreadSafeConversationStore] = None
_STORE_LOCK = threading.Lock()


def get_conversation_store() -> ThreadSafeConversationStore:
    """获取全局线程安全存储（懒加载单例）"""
    global _THREAD_SAFE_STORE
    if _THREAD_SAFE_STORE is None:
        with _STORE_LOCK:
            if _THREAD_SAFE_STORE is None:
                _THREAD_SAFE_STORE = ThreadSafeConversationStore()
    return _THREAD_SAFE_STORE


__all__ = [
    "ThreadSafeConversationStore",
    "get_conversation_store",
]
