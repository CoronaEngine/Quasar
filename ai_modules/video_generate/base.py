from __future__ import annotations


from typing import Any, Dict
import logging

from ...ai_config.ai_config import get_ai_config

from ...ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    session_context,
    extract_parameter,
    parse_tool_response,
)
from ...ai_tools.concurrency import session_concurrency
from ...ai_service.entrance import register_entrance
from ...ai_tools.helpers import request_time_diff
from ...ai_tools.request_parser import (
    extract_prompt_from_llm_content,
    extract_images_from_request,
)

from ...ai_tools.session_tracking import (
    init_session,
    update_session_state,
    set_session_error,
)

logger = logging.getLogger(__name__)


def _clean_video_parts(original_parts: list[dict]) -> list[dict]:
    """清洗视频生成返回的 parts

    Args:
        original_parts: 原始 parts 列表

    Returns:
        清洗后的 parts 列表
    """
    cleaned_parts = []
    for part in original_parts:
        cleaned_part = {
            "content_type": part.get("content_type"),
            "content_url": part.get("content_url"),
            "content_text": part.get("content_text", ""),
        }
        # 保留 url_expire_time 字段
        if "url_expire_time" in part:
            cleaned_part["url_expire_time"] = part["url_expire_time"]
        # 严格过滤 parameter
        if "parameter" in part:
            original_param = part["parameter"]
            cleaned_param = {}
            if "resolution" in original_param:
                cleaned_param["resolution"] = original_param["resolution"]
            if "duration" in original_param:
                cleaned_param["duration"] = original_param["duration"]
            if cleaned_param:
                cleaned_part["parameter"] = cleaned_param

        # 移除 None 值字段
        cleaned_part = {k: v for k, v in cleaned_part.items() if v is not None}
        cleaned_parts.append(cleaned_part)
    return cleaned_parts


@register_entrance(handler_name="handle_video_generation")
def handle_video_generation(payload: Any) -> str:
    """视频生成三层结构。"""
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    session_id = request_data.get("session_id") or "default"
    metadata = request_data.get("metadata", {})
    cfg = get_ai_config()

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type="video",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
        return _handle_video_generation_inner(request_data, session_id, metadata, cfg)


def _handle_video_generation_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    """视频生成内部实现（在并发控制内执行）"""
    try:
        # 初始化会话追踪
        init_session(
            session_id=session_id,
            input_type="video",
            parameters=request_data,
        )
        update_session_state(session_id, "running")

        logger.debug(f"收到视频生成请求: {request_data}")
        prompt = extract_prompt_from_llm_content(request_data)
        image_urls = extract_images_from_request(request_data)
        image_url = image_urls[0] if image_urls else ""

        if not image_url:
            raise ValueError("缺少 prompt 或 image_url")
        elif not prompt:
            prompt = "生成一个图片相关的视频"

        resolution = extract_parameter(request_data, "resolution", "720P")
        prompt_extend = extract_parameter(request_data, "prompt_extend", True)
        logger.debug(
            f"视频生成参数: prompt={prompt}, image_url={image_url}, resolution={resolution},\
 prompt_extend={prompt_extend}"
        )

        from .tools.video_tools import (
            load_video_tools,
        )

        tools = load_video_tools(cfg)
        if not tools:
            raise RuntimeError("视频生成功能未启用或配置不完整")
        video_tool = tools[0]
        with session_context(session_id) as sid:
            logger.debug(f"进入 session_context: {sid}")
            result_json = video_tool.func(
                prompt=prompt,
                image_url=image_url,
                resolution=resolution,
                prompt_extend=prompt_extend,
            )
            session_id = sid
        logger.debug(f"video_tool 返回: {result_json}")

        # 解析 Tool 返回的 Envelope JSON
        tool_envelope = parse_tool_response(result_json)
        logger.debug(f"解析 tool_envelope: {tool_envelope}")

        # 检查错误
        if tool_envelope.get("error_code", 0) != 0:
            error_msg = tool_envelope.get("status_info", "未知错误")
            logger.error(f"视频生成失败: {error_msg}")
            raise RuntimeError(f"视频生成失败: {error_msg}")

        # 提取 llm_content
        llm_content = tool_envelope.get("llm_content", [])
        if not llm_content:
            logger.error("视频生成未返回有效内容")
            raise RuntimeError("视频生成未返回有效内容")

        # 提取并清洗 parts
        original_parts = llm_content[0].get("part", [])
        cleaned_parts = _clean_video_parts(original_parts)
        logger.debug(f"清洗后的 parts: {cleaned_parts}")

        if not cleaned_parts:
            logger.error("视频生成未返回有效的视频部分")
            raise RuntimeError("视频生成未返回有效的视频部分")

        # 解析 parts 中的 fileid:// URL（返回真实 OSS URL 给用户）
        from ...ai_tools.response_adapter import (
            resolve_parts,
        )

        try:
            cleaned_parts = resolve_parts(cleaned_parts, timeout=150.0)
            logger.debug(f"解析后的 parts: {cleaned_parts}")
        except Exception as e:
            logger.error(f"解析 parts 中的 file_id 失败: {e}")
            # 解析失败时抛出异常，触发错误响应
            raise RuntimeError(f"视频资源解析失败: {e}") from e

        update_session_state(session_id, "completed")

        return build_success_response(
            interface_type="video",
            session_id=session_id,
            metadata=metadata,
            parts=cleaned_parts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"视频生成异常: {exc}")
        set_session_error(session_id, str(exc))
        update_session_state(session_id, "failed")
        return build_error_response(
            interface_type="video",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )


__all__ = ["handle_video_generation"]
