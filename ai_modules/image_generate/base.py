from __future__ import annotations


from typing import Any, Dict
import logging

from ai_config.ai_config import get_ai_config

from ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    extract_parameter,
    parse_tool_response,
)
from ai_tools.concurrency import session_concurrency
from ai_service.entrance import register_entrance
from ai_tools.helpers import request_time_diff
from ai_tools.request_parser import (
    normalize_image_size,
    extract_prompt_from_llm_content,
    extract_images_from_request,
)
from ai_tools.session_tracking import (
    init_session,
    update_session_state,
    set_session_error,
)
from ai_tools.workflow_executor import (
    extract_function_id,
    execute_workflow,
)

logger = logging.getLogger(__name__)


def _clean_image_parts(original_parts: list[dict]) -> list[dict]:
    """清洗图像生成返回的 parts

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
            if "image_size" in original_param:
                cleaned_param["image_size"] = normalize_image_size(
                    original_param["image_size"]
                )
            if cleaned_param:
                cleaned_part["parameter"] = cleaned_param

        # 移除 None 值字段
        cleaned_part = {k: v for k, v in cleaned_part.items() if v is not None}
        # 当 content_text 为空字符串时移除该键，避免噪声
        if "content_text" in cleaned_part and cleaned_part["content_text"] == "":
            cleaned_part.pop("content_text", None)
        cleaned_parts.append(cleaned_part)
    return cleaned_parts


@register_entrance(handler_name="handle_image_generation")
def handle_image_generation(payload: Any) -> str:
    """图像生成，返回三层结构。

    路由逻辑：
    - 如果请求包含 function_id → 调用对应的工作流
    - 如果没有 function_id → 使用直接工具调用模式
    """
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    session_id = request_data.get("session_id") or "default"
    metadata = request_data.get("metadata", {})
    cfg = get_ai_config()
    # 提取 function_id，决定是否使用工作流
    function_id = extract_function_id(request_data)

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type="image",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )

        # 根据是否有 function_id 选择处理路径
        if function_id is not None:
            return execute_workflow(
                request_data=request_data,
                interface_type="image",
            )
        else:
            return _handle_image_generation_inner(
                request_data, session_id, metadata, cfg
            )


def _handle_image_generation_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    """图像生成内部实现（在并发控制内执行）"""
    try:
        # 初始化会话追踪
        init_session(
            session_id=session_id,
            input_type="image",
            parameters=request_data,
        )
        update_session_state(session_id, "running")

        logger.debug(f"收到图像生成请求: {request_data}")
        prompt = extract_prompt_from_llm_content(request_data)
        if not prompt:
            raise ValueError("缺少图像生成的 prompt")

        from ai_modules.image_generate.tools.image_tools import (
            load_image_tools,
        )

        tools = load_image_tools(cfg)
        if not tools:
            raise RuntimeError("图像生成功能未启用或配置不完整")

        image_tool = tools[0]

        # 提取图片 URL 列表
        image_urls = extract_images_from_request(request_data)
        logger.debug(f"提取到图片 URL 列表: {image_urls}")

        # 提取 resolution 参数（图片比例）
        resolution = extract_parameter(request_data, "resolution", "1:1")
        logger.debug(f"使用 resolution: {resolution}")

        # 提取 image_size 参数（分辨率档位：1K/2K/4K，默认 2K）
        image_size = normalize_image_size(
            extract_parameter(request_data, "image_size", "2K")
        )
        # 确保有默认值
        if not image_size:
            image_size = "2K"
        logger.debug(f"使用 image_size: {image_size}")

        result_json = image_tool.invoke(
            {
                "prompt": prompt,
                "resolution": resolution,
                "image_urls": image_urls if image_urls else None,
                "image_size": image_size,
            },
            config={"session_id": session_id},
        )

        logger.debug(f"image_tool 返回: {result_json}")

        # 解析 Tool 返回的 Envelope JSON
        tool_envelope = parse_tool_response(result_json)
        logger.debug(f"解析 tool_envelope: {tool_envelope}")

        # 检查错误
        if tool_envelope.get("error_code", 0) != 0:
            error_msg = tool_envelope.get("status_info", "未知错误")
            logger.error(f"图像生成失败: {error_msg}")
            raise RuntimeError(f"图像生成失败: {error_msg}")

        # 提取 llm_content
        llm_content = tool_envelope.get("llm_content", [])
        if not llm_content:
            logger.error("图像生成未返回有效内容")
            raise RuntimeError("图像生成未返回有效内容")

        # 提取并清洗 parts
        original_parts = llm_content[0].get("part", [])
        cleaned_parts = _clean_image_parts(original_parts)
        logger.debug(f"清洗后的 parts: {cleaned_parts}")

        if not cleaned_parts:
            logger.error("图像生成未返回有效的图片部分")
            raise RuntimeError("图像生成未返回有效的图片部分")

        # 解析 parts 中的 fileid:// URL（返回真实 OSS URL 给用户）
        from ai_tools.response_adapter import (
            resolve_parts,
        )

        try:
            cleaned_parts = resolve_parts(cleaned_parts, timeout=150.0)
            logger.debug(f"解析后的 parts: {cleaned_parts}")
        except Exception as e:
            logger.error(f"解析 parts 中的 file_id 失败: {e}")
            # 解析失败时抛出异常，触发错误响应
            raise RuntimeError(f"图像资源解析失败: {e}") from e

        # 任务完成，更新状态
        update_session_state(session_id, "completed")

        return build_success_response(
            interface_type="image",
            session_id=session_id,
            metadata=metadata,
            parts=cleaned_parts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"图像生成异常: {exc}")
        # 记录错误并更新状态
        set_session_error(session_id, str(exc))
        update_session_state(session_id, "failed")
        return build_error_response(
            interface_type="image",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )


__all__ = ["handle_image_generation"]
