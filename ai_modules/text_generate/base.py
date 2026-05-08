from __future__ import annotations


from typing import Any, Dict
import logging

from ...ai_agent.conversation import default_session_id
from ...ai_config.ai_config import get_ai_config

from ...ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    pick_tool,
    session_context,
    extract_parameter,
    parse_tool_response,
)
from ...ai_tools.concurrency import session_concurrency
from ...ai_service.entrance import register_entrance
from ...ai_tools.helpers import request_time_diff
from ...ai_tools.request_parser import extract_prompt_from_llm_content
from ...ai_tools.session_tracking import (
    init_session,
    update_session_state,
    set_session_error,
)

logger = logging.getLogger(__name__)


# 使用基类提供的 extract_prompt_from_llm_content


def _clean_text_parts(original_parts: list[dict]) -> list[dict]:
    """清洗文本生成返回的 parts

    Args:
        original_parts: 原始 parts 列表

    Returns:
        清洗后的 parts 列表
    """
    cleaned_parts = []
    for part in original_parts:
        cleaned_part = {
            "content_type": part.get("content_type"),
            "content_text": part.get("content_text", ""),
        }
        # 严格过滤 parameter
        if "parameter" in part:
            original_param = part["parameter"]
            cleaned_param = {}
            if "text_type" in original_param:
                cleaned_param["text_type"] = original_param["text_type"]
            if cleaned_param:
                cleaned_part["parameter"] = cleaned_param

        # 移除 None 值字段
        cleaned_part = {k: v for k, v in cleaned_part.items() if v is not None}
        cleaned_parts.append(cleaned_part)
    return cleaned_parts


@register_entrance(handler_name="handle_text_generation")
def handle_text_generation(payload: Any) -> str:
    """文本生成三层结构。"""
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    metadata = request_data.get("metadata", {})
    session_id = request_data.get("session_id", default_session_id())
    cfg = get_ai_config()

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type="text",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
        return _handle_text_generation_inner(request_data, session_id, metadata, cfg)


def _handle_text_generation_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    """文本生成内部实现（在并发控制内执行）"""
    try:
        # 初始化会话追踪
        init_session(
            session_id=session_id,
            input_type="text",
            parameters=request_data,
        )
        update_session_state(session_id, "running")

        logger.debug(f"收到文本生成请求: {request_data}")
        text_type = extract_parameter(request_data, "text_type", "product")
        if text_type not in ["product", "marketing", "creative"]:
            raise ValueError(f"不支持的文案类型: {text_type}")

        from .tools.text_tools import load_text_tools

        tools = load_text_tools(cfg)
        if not tools:
            raise RuntimeError("文案生成功能未启用或配置不完整")
        tool_map = {
            "product": ["generate_product_text"],
            "marketing": ["generate_marketing_text"],
            "creative": ["generate_creative_text"],
        }
        text_tool = pick_tool(tools, tool_map[text_type])

        instruction = extract_prompt_from_llm_content(request_data)
        if not instruction and "message" in request_data:
            instruction = request_data["message"]

        if not instruction or not instruction.strip():
            raise ValueError("缺少文本生成的指令内容")

        # 提取通用参数
        style = extract_parameter(request_data, "style")
        length = extract_parameter(request_data, "length")

        tool_params: Dict[str, Any] = {"instruction": instruction.strip()}
        if style:
            tool_params["style"] = style
        if length:
            tool_params["length"] = length

        # 提取特定参数
        if text_type == "marketing":
            platform = extract_parameter(request_data, "platform")
            tone = extract_parameter(request_data, "tone")
            if platform:
                tool_params["platform"] = platform
            if tone:
                tool_params["tone"] = tone
        logger.debug(f"text_tool 参数: {tool_params}")

        with session_context(session_id) as sid:
            logger.debug(f"进入 session_context: {sid}")
            result_json = text_tool.func(**tool_params)
            session_id = sid
        logger.debug(f"text_tool 返回: {result_json}")

        # 解析 Tool 返回的 Envelope JSON
        tool_envelope = parse_tool_response(result_json)
        logger.debug(f"解析 tool_envelope: {tool_envelope}")

        # 检查错误
        if tool_envelope.get("error_code", 0) != 0:
            error_msg = tool_envelope.get("status_info", "未知错误")
            logger.error(f"文案生成失败: {error_msg}")
            raise RuntimeError(f"文案生成失败: {error_msg}")

        # 提取 llm_content
        llm_content = tool_envelope.get("llm_content", [])
        if not llm_content:
            logger.error("文案生成未返回有效内容")
            raise RuntimeError("文案生成未返回有效内容")

        # 提取并清洗 parts
        original_parts = llm_content[0].get("part", [])
        cleaned_parts = _clean_text_parts(original_parts)
        logger.debug(f"清洗后的 parts: {cleaned_parts}")

        if not cleaned_parts:
            logger.error("文案生成未返回有效的文本部分")
            raise RuntimeError("文案生成未返回有效的文本部分")

        # 解析 parts 中的 fileid:// URL（如有媒体内容）
        from ...ai_tools.response_adapter import (
            resolve_parts,
        )

        try:
            cleaned_parts = resolve_parts(cleaned_parts, timeout=150.0)
            logger.debug(f"解析后的 parts: {cleaned_parts}")
        except Exception as e:
            logger.error(f"解析 parts 中的 file_id 失败: {e}")
            # 解析失败时抛出异常，触发错误响应
            raise RuntimeError(f"文本资源解析失败: {e}") from e

        update_session_state(session_id, "completed")

        return build_success_response(
            interface_type="text",
            session_id=session_id,
            metadata=metadata,
            parts=cleaned_parts,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001        set_session_error(session_id, str(exc))
        update_session_state(session_id, "failed")
        logger.error(f"文本生成异常: {exc}")
        set_session_error(session_id, str(exc))
        update_session_state(session_id, "failed")
        return build_error_response(
            interface_type="text",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )


__all__ = ["handle_text_generation"]
