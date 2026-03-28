from __future__ import annotations

from typing import List, Sequence
from langchain_core.messages import BaseMessage

from ai_tools.context import (
    get_boot_session_id,
    get_current_session,
)
from ai_agent.conversation_store import (
    get_conversation_store,
)


def get_history(session_id: str) -> List[BaseMessage]:
    """获取会话历史（线程安全）"""
    return get_conversation_store().snapshot(session_id)


def update_history(session_id: str, messages: Sequence[BaseMessage]) -> None:
    """更新会话历史（线程安全）"""
    get_conversation_store().update(session_id, messages)


def default_session_id() -> str:
    return get_current_session() or get_boot_session_id()


__all__ = ["get_history", "update_history", "default_session_id"]
