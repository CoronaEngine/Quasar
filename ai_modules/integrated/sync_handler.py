from __future__ import annotations

import json
import logging

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ai_agent.executor import fallback_completion
from ai_agent.interface import process_chat_request
from ai_tools.common import build_error_response

from .workflow_bridge import resolve_and_run_workflow

from .common import (
    ToolError,
    build_success_chunk,
    build_tool_error_entry,
    extract_text_parts,
    make_assistant_entry,
    resolve_tool_message,
)

logger = logging.getLogger(__name__)


def handle_integrated_entrance_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
) -> str:
    """统一聊天接口内部实现。"""
    try:
        logger.info("收到 integrated 请求")
        logger.debug(f"请求详情: {request_data}")
        llm_content = request_data.get("llm_content", [])
        if not isinstance(llm_content, list) or not llm_content:
            raise ValueError("llm_content 不能为空")

        workflow_response = resolve_and_run_workflow(
            request_data,
            interface_type="integrated",
        )
        if workflow_response is not None:
            logger.info("integrated 请求命中 workflow 路由")
            logger.debug(f"工作流返回: {workflow_response[:200]}...")  # 打印前200字符
            
            # 验证返回值是否是有效的 JSON
            try:
                response_data = json.loads(workflow_response)
                if response_data.get("error_code") == 0:
                    logger.info("工作流执行成功，返回结果到前端")
                    return workflow_response
                else:
                    error_code = response_data.get("error_code", -1)
                    status_info = response_data.get("status_info", "未知错误")
                    logger.error(f"工作流返回错误: error_code={error_code}, status_info={status_info}")
                    # 即使是错误也应该返回给前端
                    return workflow_response
            except json.JSONDecodeError as parse_err:
                logger.error(f"工作流返回的响应不是有效 JSON: {parse_err}")
                # 即使不是 JSON，如果返回了内容，也直接返回
                return workflow_response
        else:
            logger.debug("工作流未命中，继续走 Agent 路由")

        result = process_chat_request(request_data)
        logger.debug(f"Agent 运行结果: {result}")
        messages: List[BaseMessage] = result["messages"]
        session_id = result["session_id"]
        pending_history = result["pending_history"]

        start_index = 0
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                start_index = index + 1
                break

        relevant_messages = messages[start_index:]
        logger.debug(f"相关消息: {relevant_messages}")

        current_entry: Optional[Dict[str, Any]] = None

        for msg in relevant_messages:
            if isinstance(msg, AIMessage):
                text_parts = extract_text_parts(msg)
                if text_parts:
                    if current_entry is None:
                        current_entry = make_assistant_entry(text_parts)
                        logger.debug(f"创建 entry（AIMessage）: {current_entry}")
                    else:
                        current_entry["part"].extend(text_parts)
                        logger.debug("追加 AIMessage text parts 到 entry")

            elif isinstance(msg, ToolMessage):
                tool_parts = resolve_tool_message(msg)
                if tool_parts is None:
                    current_entry = build_tool_error_entry(
                        ToolError("工具服务暂时不可用，请稍后重试。")
                    )
                    continue

                if tool_parts:
                    if current_entry is None:
                        current_entry = make_assistant_entry(tool_parts)
                        logger.debug(f"创建 entry（ToolMessage）: {current_entry}")
                    else:
                        current_entry["part"].extend(tool_parts)
                        logger.debug(f"追加 tool parts 到 entry: {tool_parts}")

        if current_entry is None:
            logger.warning("current_entry 为空，进入 fallback 处理")
            fallback_text = fallback_completion(pending_history)
            if not fallback_text:
                logger.error("Agent 未返回有效响应，fallback 也失败")
                raise ValueError("Agent 未返回有效响应")
            current_entry = make_assistant_entry([
                {
                    "content_type": "text",
                    "content_text": fallback_text,
                    "content_url": "",
                    "parameter": {},
                }
            ])
            logger.debug(f"添加 fallback entry: {current_entry}")

        response = build_success_chunk(session_id, metadata, current_entry)
        logger.info("integrated 请求处理完成")
        logger.debug(f"返回成功响应: {response}")
        return response

    except Exception as exc:
        session_id = request_data.get("session_id", "default")
        logger.error(f"处理 integrated 入口异常: {exc}")
        return build_error_response(
            interface_type="integrated",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )


__all__ = ["handle_integrated_entrance_inner"]
