# file: Backend/artificial_intelligence/agent/protocol.py

from __future__ import annotations
import json

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, ToolMessage


def extract_session_id(payload: Any, default_session: str) -> str:
    if isinstance(payload, dict):
        return str(payload.get("session_id") or default_session)
    return default_session


def extract_user_parts(payload: Any) -> List[Dict[str, Any]]:
    """提取用户输入中的 part 列表 (保持原逻辑)"""
    if isinstance(payload, dict) and "llm_content" in payload:
        llm_content = payload.get("llm_content", [])
        if isinstance(llm_content, list):
            for content in reversed(llm_content):
                if content.get("role") == "user":
                    return list(content.get("part", []))
    if isinstance(payload, str):
        return [{"content_type": "text", "content_text": payload}]
    return []


def wrap_part_as_assistant_message(part: Dict[str, Any]) -> AIMessage:
    """
    将 part 封装为助手消息，用于向 LLM 传递用户上传的媒体资源。

    Args:
        part: 媒体资源部分

    Returns:
        AIMessage: 包含媒体信息的助手消息
    """
    content_type = part.get("content_type", "unknown")
    content_url = part.get("content_url") or part.get("content_file", "")
    content_text = part.get("content_text", "")

    # 构建简洁的文本描述
    description = f"[用户上传了{content_type}] URL: {content_url}"
    if content_text:
        description += f" 描述: {content_text}"

    return AIMessage(content=description)


def extract_assistant_messages(messages: List[Any]) -> List[AIMessage]:
    result: List[AIMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            result.append(msg)
        elif isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            result.append(AIMessage(content=content))
    return result


def extract_tool_media_parts(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    从 Agent 返回的 messages 中提取工具产生的媒体 part

    遍历所有 ToolMessage，解析其 content（JSON 格式），
    提取 llm_content[].part[] 中 content_type 为 image/video/audio 的 part。

    Args:
        messages: Agent 返回的消息列表

    Returns:
        媒体 part 列表
    """
    media_parts: List[Dict[str, Any]] = []

    for msg in messages:
        # 只处理 ToolMessage
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content
        if not isinstance(content, str):
            continue

        try:
            envelope = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue

        # 解析 llm_content
        llm_content = envelope.get("llm_content", [])
        if not isinstance(llm_content, list):
            continue

        for content_block in llm_content:
            parts = content_block.get("part", [])
            if not isinstance(parts, list):
                continue

            for part in parts:
                content_type = part.get("content_type")
                if content_type in ("image", "video", "audio"):
                    media_parts.append(part)

    return media_parts


def build_media_history_assistant_message(
    parts: List[Dict[str, Any]],
    session_id: str,
) -> AIMessage:
    """
    将媒体历史记录构造为助手消息，用于向 LLM 传递历史媒体资源。

    Args:
        parts: 媒体 part 列表
        session_id: 会话 ID

    Returns:
        AIMessage: 包含媒体历史信息的助手消息
    """
    # 按类型分组统计
    type_counts: Dict[str, int] = {}
    for part in parts:
        ct = part.get("content_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1

    summary = ", ".join([f"{count}个{t}" for t, count in type_counts.items()])

    # 构建媒体列表描述
    media_list = []
    for i, part in enumerate(parts, 1):
        ct = part.get("content_type", "unknown")
        url = part.get("content_url", "")
        text = part.get("content_text", "")
        item = f"{i}. [{ct}] {url}"
        if text:
            item += f" - {text}"
        media_list.append(item)

    content = f"[历史媒体资源汇总: {summary}]\n" + "\n".join(media_list)

    return AIMessage(content=content)


__all__ = [
    "extract_session_id",
    "extract_user_parts",
    "wrap_part_as_assistant_message",
    "extract_assistant_messages",
    "extract_tool_media_parts",
    "build_media_history_assistant_message",
]
