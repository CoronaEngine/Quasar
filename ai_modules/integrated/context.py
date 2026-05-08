from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def prepare_stream_context(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备流式 Agent 所需的上下文：解析 session、history、用户媒体，构建 pending_history。
    """
    from ...ai_agent.protocol import (
        build_media_history_assistant_message,
        extract_session_id,
        extract_user_parts,
        wrap_part_as_assistant_message,
    )
    from ...ai_agent.conversation import default_session_id, get_history
    from ...ai_media_resource import get_media_registry

    session_id = extract_session_id(request_data, default_session_id())
    stored_history = get_history(session_id)
    raw_parts = extract_user_parts(request_data)
    media_registry = get_media_registry()

    human_content_blocks: List[Dict[str, Any]] = []
    artificial_assistant_messages: List[AIMessage] = []
    uploaded_media_parts: List[Dict[str, Any]] = []

    for part in raw_parts:
        content_type = part.get("content_type")
        if content_type == "text":
            text = part.get("content_text", "").strip()
            if text:
                human_content_blocks.append({"type": "text", "text": text})
        elif content_type in ["image", "video", "audio", "file"]:
            if part.get("content_url"):
                uploaded_media_parts.append(part)

    if uploaded_media_parts:
        file_ids = media_registry.register_batch(session_id, uploaded_media_parts)
        for part, file_id in zip(uploaded_media_parts, file_ids):
            content_type = part.get("content_type")
            clean_part = {
                "content_type": content_type,
                "content_url": f"fileid://{file_id}",
                "content_text": part.get("content_text", ""),
                "content_file": part.get("content_url", ""),
            }
            if "parameter" in part:
                clean_part["parameter"] = part["parameter"]
            artificial_assistant_messages.append(wrap_part_as_assistant_message(clean_part))
            if human_content_blocks:
                human_content_blocks[0]["text"] += f"[PS: 前文包含上传的{content_type}信息。]"

    media_history_parts = media_registry.get_session_parts(session_id, resolved_only=True)
    media_history_message: List[AIMessage] = []
    if media_history_parts:
        media_history_message = [
            build_media_history_assistant_message(media_history_parts, session_id)
        ]

    if not human_content_blocks:
        human_content_blocks.append({"type": "text", "text": "[Attachment Uploaded]"})

    pending_history: List[BaseMessage] = [
        *stored_history,
        *media_history_message,
        *artificial_assistant_messages,
        HumanMessage(content=human_content_blocks),
    ]

    return {
        "session_id": session_id,
        "pending_history": pending_history,
        "media_registry": media_registry,
    }


__all__ = ["prepare_stream_context"]
