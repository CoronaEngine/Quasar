# file: Backend/artificial_intelligence/agent/interface.py

from __future__ import annotations
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage

from ai_agent.executor import (
    run_agent,
    fallback_completion,
)
from ai_media_resource import get_media_registry
from ai_agent.conversation import (
    default_session_id,
    get_history,
    update_history,
)
from ai_agent.protocol import (
    extract_session_id,
    extract_assistant_messages,
    extract_user_parts,
    extract_tool_media_parts,
    wrap_part_as_assistant_message,
    build_media_history_assistant_message,
)
from ai_tools.context import (
    reset_current_session,
    set_current_session,
)


def process_chat_request(payload: Any) -> Dict[str, Any]:
    # 1. 提取 Session ID
    session_id = extract_session_id(payload, default_session_id())
    stored_history = get_history(session_id)

    # 2. 获取原始输入 parts
    raw_parts = extract_user_parts(payload)

    # 3. 获取媒体资源注册表
    media_registry = get_media_registry()

    human_content_blocks: List[Dict[str, Any]] = []
    artificial_assistant_messages: List[AIMessage] = []
    uploaded_media_parts: List[Dict[str, Any]] = []  # 收集用户上传的媒体（原始数据）

    # 4. 分流处理：先收集所有媒体资源
    for part in raw_parts:
        c_type = part.get("content_type")

        if c_type == "text":
            text = part.get("content_text", "").strip()
            if text:
                human_content_blocks.append({"type": "text", "text": text})

        elif c_type in ["image", "video", "audio"]:
            url = part.get("content_url")
            if url:
                # 收集上传的媒体资源（保持原始数据不变）
                uploaded_media_parts.append(part)

    # 5. 先注册用户上传的媒体，分配 file_id
    if uploaded_media_parts:
        file_ids = media_registry.register_batch(session_id, uploaded_media_parts)

        # 6. 构造助手消息：创建 part 副本，使用 fileid:// 格式
        # 这样可以避免原地修改 part 导致真实 URL 泄漏到历史
        for part, file_id in zip(uploaded_media_parts, file_ids):
            c_type = part.get("content_type")

            # 创建干净的 part 副本，只包含 fileid:// URL
            clean_part = {
                "content_type": c_type,
                "content_url": f"fileid://{file_id}",
                "content_text": part.get("content_text", ""),
            }
            if "parameter" in part:
                clean_part["parameter"] = part["parameter"]

            # 构造助手消息传递媒体信息
            assistant_msg = wrap_part_as_assistant_message(clean_part)
            artificial_assistant_messages.append(assistant_msg)

            # 确保有文本块存在再追加提示
            if human_content_blocks:
                human_content_blocks[0][
                    "text"
                ] += f"[PS: 前文包含上传的{c_type}信息。如果有工具需要使用这个{c_type}，\
请从前文的助手消息中获取对应的URL。]"

    # 7. 获取历史媒体资源（不含当前轮次），构建汇总助手消息
    # 使用 get_session_parts 获取已解析的媒体记录
    media_history_parts = media_registry.get_session_parts(
        session_id, resolved_only=True
    )
    media_history_message: List[AIMessage] = []
    if media_history_parts:
        media_history_message = [
            build_media_history_assistant_message(media_history_parts, session_id)
        ]

    if not human_content_blocks:
        human_content_blocks.append({"type": "text", "text": "[Attachment Uploaded]"})

    current_human_message = HumanMessage(content=human_content_blocks)

    # 8. 构建待处理历史：历史对话 + 媒体历史汇总 + 当前上传 + 当前消息
    pending_history = [
        *stored_history,
        *media_history_message,
        *artificial_assistant_messages,
        current_human_message,
    ]

    token = set_current_session(session_id)
    try:
        state = run_agent(pending_history)
    finally:
        reset_current_session(token)

    messages = state.get("messages", [])

    # 9. 提取工具产生的媒体资源并注册到媒体注册表
    # 工具产生的媒体已经通过 submit() 注册，这里只需要确保它们被正确追踪
    tool_media_parts = extract_tool_media_parts(messages)
    if tool_media_parts:
        # 工具产生的媒体如果尚未注册，这里进行注册
        # 注意：大多数情况下工具已经通过 submit() 注册，这里是备用路径
        media_registry.register_batch(session_id, tool_media_parts)

    history_extension = extract_assistant_messages(messages)

    new_entries = [*artificial_assistant_messages, current_human_message]

    if history_extension:
        new_entries.extend(history_extension)
        update_history(session_id, [*stored_history, *new_entries])
    else:
        fallback_text = fallback_completion(pending_history)
        if fallback_text:
            history_extension = [AIMessage(content=fallback_text)]
            new_entries.extend(history_extension)
            update_history(session_id, [*stored_history, *new_entries])

    return {
        "messages": messages,
        "session_id": session_id,
        "pending_history": pending_history,
    }


__all__ = ["process_chat_request"]
