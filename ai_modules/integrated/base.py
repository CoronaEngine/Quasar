# file: Backend/artificial_intelligence/service/base.py

from __future__ import annotations


import json
import time
import logging

from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, HumanMessage

from ai_agent.executor import (
    fallback_completion,
    stream_agent,
)
from ai_agent.interface import process_chat_request
from ai_workflow.executor import run_workflow_from_request, stream_workflow_from_request
from ai_config.ai_config import get_ai_config
from ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
)
from ai_tools.concurrency import session_concurrency
from ai_service.entrance import register_entrance
from ai_tools.helpers import request_time_diff

logger = logging.getLogger(__name__)

# 流式输出配置：每个 Assistant 消息最多累积的工具结果数量
# 超过此数量后，会自动创建新的 Assistant 消息来承载后续工具结果
MAX_TOOLS_PER_ASSISTANT = 3


class ToolError(Exception):
    """工具执行错误，用于传递工具返回的错误信息。"""

    def __init__(self, message: str, error_code: int = 1):
        super().__init__(message)
        self.error_code = error_code


def _parse_tool_parts(content: str) -> List[Dict[str, Any]]:
    """
    [修改] 解析工具返回的完整 API 响应 Envelope。
    现在支持提取所有类型的 part (包括 text)，不仅仅是媒体。

    如果工具返回了 error_code != 0 的响应，将抛出 ToolError 异常。
    """
    found_parts = []
    try:
        # 1. 清洗 content
        clean_content = content.strip()
        if clean_content.startswith("```"):
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            else:
                clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]

        # 2. 解析 JSON
        data = json.loads(clean_content.strip())
        logger.debug(f"解析工具内容成功: {data}")

        # 3. 检查错误码 - 如果工具返回错误，抛出异常
        if isinstance(data, dict):
            error_code = data.get("error_code", 0)
            if error_code != 0:
                error_msg = data.get("status_info", "工具执行失败")
                logger.error(f"工具返回错误: error_code={error_code}, status_info={error_msg}")
                raise ToolError(error_msg, error_code)

        # 4. 钻取: llm_content -> list -> part -> list
        if isinstance(data, dict):
            llm_content = data.get("llm_content", [])
            if isinstance(llm_content, list):
                for item in llm_content:
                    parts = item.get("part", [])
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict):
                                c_type = p.get("content_type")
                                if c_type in ["image", "audio", "video", "text", "file"]:
                                    found_parts.append(p)
    except ToolError:
        # 重新抛出 ToolError，不要被下面的通用异常捕获
        raise
    except Exception as e:
        logger.error(f"工具内容解析失败: {e} 内容: {content}")
    return found_parts


def _extract_text_parts(msg: AIMessage) -> List[Dict[str, Any]]:
    """从 AIMessage 提取纯文本部分"""
    parts: List[Dict[str, Any]] = []
    content_str = ""

    if isinstance(msg.content, str):
        content_str = msg.content
    elif isinstance(msg.content, list):
        texts = []
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        content_str = "\n".join(texts)

    if content_str:
        parts.append(
            {
                "content_type": "text",
                "content_text": content_str,
                "content_url": "",
                "parameter": {},
            }
        )
    logger.debug(f"提取文本部分: {parts}")
    return parts

@register_entrance(handler_name="handle_integrated_entrance")
def handle_integrated_entrance(payload: Any) -> str:
    """
    统一聊天接口
    执行逻辑：Assistant (Thought) -> Tool (Result) [挂载到前者]
    特殊情况：Tool (Text) -> 独立消息
    """
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    metadata = request_data.get("metadata", {})
    session_id = request_data.get("session_id", "default")
    cfg = get_ai_config()

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type="integrated",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
        return _handle_integrated_entrance_inner(request_data, session_id, metadata)


def _handle_integrated_entrance_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
) -> str:
    """统一聊天接口内部实现（在并发控制内执行）"""
    try:
        logger.info("收到 integrated 请求")
        logger.debug(f"请求详情: {request_data}")
        llm_content = request_data.get("llm_content", [])
        if not isinstance(llm_content, list) or not llm_content:
            raise ValueError("llm_content 不能为空")

        workflow_response = run_workflow_from_request(
            request_data,
            interface_type="integrated",
        )
        if workflow_response is not None:
            logger.info("integrated 请求命中 workflow 路由")
            return workflow_response

        # 运行 Agent
        result = process_chat_request(request_data)
        logger.debug(f"Agent 运行结果: {result}")
        messages: List[BaseMessage] = result["messages"]
        session_id = result["session_id"]
        pending_history = result["pending_history"]

        llm_content_list: List[Dict[str, Any]] = []

        # --- 1. 确定处理范围 ---
        start_index = 0
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                start_index = i + 1
                break

        relevant_messages = messages[start_index:]
        logger.debug(f"相关消息: {relevant_messages}")

        # --- 2. 核心处理逻辑 ---
        last_assistant_entry: Optional[Dict[str, Any]] = None

        for msg in relevant_messages:
            if isinstance(msg, AIMessage):
                text_parts = _extract_text_parts(msg)
                if text_parts:
                    new_entry = {
                        "role": "assistant",
                        "interface_type": "integrated",
                        "sent_time_stamp": int(time.time()),
                        "part": text_parts,
                    }
                    llm_content_list.append(new_entry)
                    last_assistant_entry = new_entry
                    logger.debug(f"添加 AIMessage entry: {new_entry}")

            elif isinstance(msg, ToolMessage):
                try:
                    tool_parts = _parse_tool_parts(msg.content)
                except ToolError as e:
                    # 工具返回了错误，记录并重新抛出以返回错误响应
                    logger.error(f"工具执行失败: {e}")
                    raise
                logger.debug(f"解析 ToolMessage 得到 parts: {tool_parts}")

                # 解析 parts 中的 fileid:// URL（返回真实 URL 给用户）
                from ai_tools.response_adapter import (
                    resolve_parts,
                )

                try:
                    tool_parts = resolve_parts(tool_parts, timeout=150.0)
                    logger.debug(f"解析后的 tool_parts: {tool_parts}")
                except Exception as e:
                    logger.error(f"解析 tool_parts 中的 file_id 失败: {e}")
                    # 解析失败时抛出异常，触发错误响应
                    raise RuntimeError(f"工具资源解析失败: {e}") from e

                for part in tool_parts:
                    c_type = part.get("content_type")
                    if c_type in ["image", "audio", "video", "file"]:
                        if last_assistant_entry is not None:
                            last_assistant_entry["part"].append(part)
                            logger.debug(f"挂载媒体 part 到上一个 entry: {part}")
                        else:
                            new_entry = {
                                "role": "assistant",
                                "interface_type": "integrated",
                                "sent_time_stamp": int(time.time()),
                                "part": [part],
                            }
                            llm_content_list.append(new_entry)
                            last_assistant_entry = new_entry
                            logger.debug(f"新建媒体 entry: {new_entry}")
                    elif c_type == "text":
                        new_entry = {
                            "role": "assistant",
                            "interface_type": "integrated",
                            "sent_time_stamp": int(time.time()),
                            "part": [part],
                        }
                        llm_content_list.append(new_entry)
                        last_assistant_entry = new_entry
                        logger.debug(f"新建文本 entry: {new_entry}")

        # --- Fallback 处理 ---
        if not llm_content_list:
            logger.warning("llm_content_list 为空，进入 fallback 处理")
            fallback_text = fallback_completion(pending_history)
            if not fallback_text:
                logger.error("Agent 未返回有效响应，fallback 也失败")
                raise ValueError("Agent 未返回有效响应")

            entry = {
                "role": "assistant",
                "interface_type": "integrated",
                "sent_time_stamp": int(time.time()),
                "part": [
                    {
                        "content_type": "text",
                        "content_text": fallback_text,
                        "content_url": "",
                        "parameter": {},
                    }
                ],
            }
            llm_content_list.append(entry)
            logger.debug(f"添加 fallback entry: {entry}")

        response = build_success_response(
            interface_type="integrated",
            session_id=session_id,
            metadata=metadata,
            llm_content=llm_content_list,
        )
        logger.info("integrated 请求处理完成")
        logger.debug(f"返回成功响应: {response}")
        return response

    except Exception as exc:
        session_id = request_data.get("session_id", "default")
        logger.error(f"处理 integrated 入口异常: {exc}")
        response = build_error_response(
            interface_type="integrated",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )
        logger.debug(f"返回错误响应: {response}")
        return response

@register_entrance(handler_name="handle_integrated_entrance_stream")
def handle_integrated_entrance_stream(payload: Any):
    """
    流式统一聊天接口
    逐步返回：每个 AIMessage + 关联的 ToolMessage 作为一个完整的响应单元

    Yields:
        每个 yield 是一个完整的 JSON 响应字符串，包含一个 llm_content entry
    """
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    metadata = request_data.get("metadata", {})
    session_id = request_data.get("session_id", "default")
    cfg = get_ai_config()

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            error_response = build_error_response(
                interface_type="integrated",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
            yield error_response
            return

        # 在并发控制内执行流式处理
        yield from _handle_integrated_entrance_stream_inner(
            request_data, session_id, metadata
        )


def _handle_integrated_entrance_stream_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
):
    """流式统一聊天接口内部实现（在并发控制内执行）"""
    try:
        logger.info("收到 integrated stream 请求")
        logger.debug(f"请求详情: {request_data}")
        llm_content = request_data.get("llm_content", [])
        if not isinstance(llm_content, list) or not llm_content:
            raise ValueError("llm_content 不能为空")

        workflow_stream = stream_workflow_from_request(
            request_data,
            interface_type="integrated",
        )
        if workflow_stream is not None:
            logger.info("integrated stream 请求命中 workflow 路由（流式）")
            yield from workflow_stream
            return

        # 获取 session_id 和准备历史
        from ai_agent.protocol import (
            extract_session_id,
            extract_user_parts,
            wrap_part_as_assistant_message,
            build_media_history_assistant_message,
            extract_tool_media_parts,
        )
        from ai_agent.conversation import (
            default_session_id,
            get_history,
            update_history,
        )
        from ai_media_resource import (
            get_media_registry,
        )

        session_id = extract_session_id(request_data, default_session_id())
        stored_history = get_history(session_id)
        raw_parts = extract_user_parts(request_data)
        media_registry = get_media_registry()

        # 构建用户消息
        human_content_blocks: List[Dict[str, Any]] = []
        artificial_assistant_messages: List[AIMessage] = []
        uploaded_media_parts: List[Dict[str, Any]] = []

        for part in raw_parts:
            c_type = part.get("content_type")
            if c_type == "text":
                text = part.get("content_text", "").strip()
                if text:
                    human_content_blocks.append({"type": "text", "text": text})
            elif c_type in ["image", "video", "audio", "file"]:
                url = part.get("content_url")
                if url:
                    # 收集上传的媒体资源（保持原始数据不变）
                    uploaded_media_parts.append(part)

        # 注册用户上传的媒体并构造助手消息
        if uploaded_media_parts:
            file_ids = media_registry.register_batch(session_id, uploaded_media_parts)

            # 构造助手消息：创建 part 副本，使用 fileid:// 格式
            for part, file_id in zip(uploaded_media_parts, file_ids):
                c_type = part.get("content_type")

                # 创建干净的 part 副本，只包含 fileid:// URL
                clean_part = {
                    "content_type": c_type,
                    "content_url": f"fileid://{file_id}",
                    "content_text": part.get("content_text", ""),
                    "content_file": part.get("content_url", ""),
                }
                if "parameter" in part:
                    clean_part["parameter"] = part["parameter"]

                assistant_msg = wrap_part_as_assistant_message(clean_part, session_id)
                artificial_assistant_messages.append(assistant_msg)

                if human_content_blocks:
                    human_content_blocks[0][
                        "text"
                    ] += f"[PS: 前文包含上传的{c_type}信息。]"

        # 获取历史媒体资源
        media_history_parts = media_registry.get_session_parts(
            session_id, resolved_only=True
        )
        media_history_message: List[AIMessage] = []
        if media_history_parts:
            media_history_message = [
                build_media_history_assistant_message(media_history_parts, session_id)
            ]

        if not human_content_blocks:
            human_content_blocks.append(
                {"type": "text", "text": "[Attachment Uploaded]"}
            )

        current_human_message = HumanMessage(content=human_content_blocks)

        # 构建完整历史
        pending_history = [
            *stored_history,
            *media_history_message,
            *artificial_assistant_messages,
            current_human_message,
        ]

        # 流式执行 Agent
        accumulated_messages = list(pending_history)
        last_assistant_entry: Optional[Dict[str, Any]] = None
        tool_count_in_current_entry = 0  # 当前 entry 累积的工具数量

        for chunk in stream_agent(pending_history):
            # chunk 格式: {"node_name": {"messages": [new_messages]}}
            for node_name, node_data in chunk.items():
                new_messages = node_data.get("messages", [])
                if not new_messages:
                    continue

                accumulated_messages.extend(new_messages)

                # 处理新消息
                for msg in new_messages:
                    if isinstance(msg, AIMessage):
                        # 新的 AIMessage 到来，先发送之前累积的内容
                        if (
                            last_assistant_entry is not None
                            and last_assistant_entry["part"]
                        ):
                            response = build_success_response(
                                interface_type="integrated",
                                session_id=session_id,
                                metadata=metadata,
                                llm_content=[last_assistant_entry],
                            )
                            logger.debug(
                                f"[Stream] 新AIMessage到来，先Yield之前累积的: {response}"
                            )
                            yield response

                        # 创建新的 assistant entry
                        text_parts = _extract_text_parts(msg)
                        if text_parts:
                            last_assistant_entry = {
                                "role": "assistant",
                                "interface_type": "integrated",
                                "sent_time_stamp": int(time.time()),
                                "part": text_parts,
                            }
                            tool_count_in_current_entry = 0  # 重置工具计数
                            logger.debug(
                                f"[Stream] 创建 AIMessage entry: {last_assistant_entry}"
                            )
                        else:
                            # AIMessage 没有文本内容（纯工具调用）
                            last_assistant_entry = None
                            tool_count_in_current_entry = 0

                    elif isinstance(msg, ToolMessage):
                        # 将工具结果附加到当前 assistant entry
                        try:
                            tool_parts = _parse_tool_parts(msg.content)
                        except ToolError as e:
                            # 工具返回了错误，记录并重新抛出以返回错误响应
                            logger.error(f"[Stream] 工具执行失败: {e}")
                            raise
                        logger.debug(f"[Stream] 解析 ToolMessage: {tool_parts}")

                        # 解析 parts 中的 fileid:// URL（流式输出也需要返回真实 URL 给用户）
                        from ai_tools.response_adapter import (
                            resolve_parts,
                        )

                        try:
                            tool_parts = resolve_parts(tool_parts, timeout=150.0)
                            logger.debug(f"[Stream] 解析后的 tool_parts: {tool_parts}")
                        except Exception as e:
                            logger.error(
                                f"[Stream] 解析 tool_parts 中的 file_id 失败: {e}"
                            )
                            # 解析失败时抛出异常，触发错误响应
                            raise RuntimeError(f"工具资源解析失败: {e}") from e

                        # 检查是否需要创建新的 entry（超过累积上限）
                        if (
                            last_assistant_entry is not None
                            and tool_count_in_current_entry >= MAX_TOOLS_PER_ASSISTANT
                        ):
                            # 先发送当前累积的内容
                            response = build_success_response(
                                interface_type="integrated",
                                session_id=session_id,
                                metadata=metadata,
                                llm_content=[last_assistant_entry],
                            )
                            logger.debug(
                                f"[Stream] 工具累积达到上限({MAX_TOOLS_PER_ASSISTANT})，Yield: {response}"
                            )
                            yield response

                            # 创建新的 entry 来承载后续工具结果
                            last_assistant_entry = {
                                "role": "assistant",
                                "interface_type": "integrated",
                                "sent_time_stamp": int(time.time()),
                                "part": [],
                            }
                            tool_count_in_current_entry = 0
                            logger.debug("[Stream] 创建新 entry 承载后续工具结果")

                        # 将工具结果附加到 entry
                        for part in tool_parts:
                            c_type = part.get("content_type")
                            if c_type in ["image", "audio", "video", "text", "file"]:
                                if last_assistant_entry is not None:
                                    last_assistant_entry["part"].append(part)
                                    logger.debug(
                                        f"[Stream] 附加工具输出到 entry: {part}"
                                    )
                                else:
                                    # 没有 assistant entry，创建一个新的
                                    last_assistant_entry = {
                                        "role": "assistant",
                                        "interface_type": "integrated",
                                        "sent_time_stamp": int(time.time()),
                                        "part": [part],
                                    }
                                    tool_count_in_current_entry = 0
                                    logger.debug(
                                        f"[Stream] 创建新 entry 承载工具结果: {part}"
                                    )

                        # 增加工具计数
                        tool_count_in_current_entry += 1

        # 如果最后还有未发送的 AIMessage
        if last_assistant_entry is not None and last_assistant_entry["part"]:
            response = build_success_response(
                interface_type="integrated",
                session_id=session_id,
                metadata=metadata,
                llm_content=[last_assistant_entry],
            )
            logger.debug(f"[Stream] Yield 最终响应: {response}")
            yield response

        # 提取工具生成的媒体并注册
        tool_media_parts = extract_tool_media_parts(accumulated_messages)
        if tool_media_parts:
            media_registry.register_batch(session_id, tool_media_parts)

        # 保存历史
        update_history(session_id, accumulated_messages)
        logger.info("integrated stream 请求处理完成")

    except Exception as exc:
        logger.error(f"处理 integrated stream 入口异常: {exc}", exc_info=True)
        error_response = build_error_response(
            interface_type="integrated",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )
        yield error_response


__all__ = ["handle_integrated_entrance", "handle_integrated_entrance_stream"]
