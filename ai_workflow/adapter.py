"""
输入输出适配器

负责将外部请求数据转换为 WorkflowState，
以及将 State 中的结果转换为标准三层响应格式。

输入适配:
- 从 llm_content 中解析 function_id、prompt、images 等字段
- 支持从 part[].parameter.function_id 或顶层 function_id 获取

输出适配:
- 解析 tool_results 中最后一个工具返回的 JSON
- 提取 parts 并调用 build_success_response 构建响应
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ai_workflow.state import (
    WorkflowState,
    create_initial_state,
)
from ai_tools.common import (
    ensure_dict,
    extract_parameter,
    build_success_response,
    build_error_response,
)

logger = logging.getLogger(__name__)


def parse_request(request_data: Any) -> WorkflowState:
    """解析请求数据为 WorkflowState

    从标准三层结构中提取工作流所需的字段。

    Args:
        request_data: 原始请求数据（可以是 dict 或 JSON 字符串）

    Returns:
        初始化的 WorkflowState

    Raises:
        ValueError: 当缺少必需字段（如 function_id）
    """
    data = ensure_dict(request_data)

    # 提取 session_id
    session_id = data.get("session_id") or "default"

    # 提取 metadata
    metadata = data.get("metadata", {})

    # 提取 function_id（优先从 parameter 中获取）
    function_id = extract_parameter(data, "function_id")
    if function_id is None:
        raise ValueError("Missing required field: function_id")

    # 确保 function_id 为 int
    if isinstance(function_id, str):
        function_id = int(function_id)

    # 提取 prompt（从 text 类型的 part 中获取）
    prompt = _extract_prompt(data)

    # 提取 images 和对应的 bounding_box（建立一一对应关系）
    images, bounding_box = _extract_images_with_bboxes(data)

    additional_type = extract_parameter(data, "additional_type", None)

    # 提取生成参数
    resolution = extract_parameter(data, "resolution", "1:1")
    image_size = _normalize_image_size(extract_parameter(data, "image_size", "2K"))

    return create_initial_state(
        session_id=session_id,
        function_id=function_id,
        prompt=prompt,
        images=images,
        additional_type=additional_type,
        bounding_box=bounding_box,
        resolution=resolution,
        image_size=image_size,
        metadata=metadata,
    )


def _extract_prompt(data: Dict[str, Any]) -> str:
    """从 llm_content 中提取文本 prompt"""
    llm_content = data.get("llm_content")
    if not isinstance(llm_content, list) or not llm_content:
        return ""

    parts = llm_content[0].get("part", [])
    prompts = []

    for part in parts:
        if part.get("content_type") == "text":
            text = part.get("content_text", "").strip()
            if text:
                prompts.append(text)

    return " ".join(prompts)


def _extract_images(data: Dict[str, Any]) -> List[str]:
    """从 llm_content 中提取图片 URL 列表

    支持 content_type 为 "image" 或 "detection" 的 part。
    """
    llm_content = data.get("llm_content")
    if not isinstance(llm_content, list) or not llm_content:
        return []

    parts = llm_content[0].get("part", [])
    images = []

    for part in parts:
        content_type = part.get("content_type")
        if content_type in ("image", "detection"):
            url = part.get("content_url")
            if url:
                images.append(url)

    return images


def _extract_images_with_bboxes(data: Dict[str, Any]) -> tuple:
    """按 part 顺序收集图片和对应的 bounding_box

    支持两种场景：
    1. detection part：多个 box（box 中可能有 url 字段用于替换）
    2. image part：单个 box（用于指定裁剪区域）

    返回：
        (images, bounding_box_list)
        其中 bounding_box_list[i] 对应 images[i] 的 box 列表
        - images: [url1, url2, ...]
        - bounding_box_list: [[box1_for_img1, ...], [box1_for_img2, ...], ...]
    """
    llm_content = data.get("llm_content")
    if not isinstance(llm_content, list) or not llm_content:
        return [], []

    parts = llm_content[0].get("part", [])
    images = []
    bounding_box_list = []

    for part in parts:
        content_type = part.get("content_type")
        if content_type not in ("image", "detection"):
            continue

        url = part.get("content_url")
        if not url:
            continue

        images.append(url)

        # 提取该 part 对应的 bounding_box
        param = part.get("parameter", {})
        bbox = param.get("bounding_box", [])

        # 统一转换为列表格式
        if isinstance(bbox, dict):
            bbox = [bbox]
        elif not isinstance(bbox, list):
            bbox = []

        bounding_box_list.append(bbox)

    return images, bounding_box_list


def _normalize_image_size(value: Optional[str]) -> str:
    """规范化 image_size（如 '1k' -> '1K'）"""
    if value is None:
        return "1K"
    v = value.strip()
    if v.lower().endswith("k"):
        return v[:-1].upper() + "K"
    return v


def format_response(
    state: WorkflowState,
    *,
    interface_type: str = "image",
) -> str:
    """将 WorkflowState 转换为标准响应 JSON

    优先使用 output_parts，若为空则从 tool_results 最后一项解析。

    Args:
        state: 工作流最终状态
        interface_type: 接口类型

    Returns:
        标准三层结构的 JSON 字符串
    """
    session_id = state.get("session_id", "default")
    metadata = state.get("metadata", {})

    # 检查错误
    error = state.get("error")
    if error:
        return build_error_response(
            interface_type=interface_type,
            session_id=session_id,
            exc=RuntimeError(error),
            metadata=metadata,
        )

    llm_content = state.get("output_llm_content", [])
    if llm_content:
        return build_success_response(
            interface_type=interface_type,
            session_id=session_id,
            metadata=metadata,
            llm_content=llm_content,
        )

    # 优先使用 output_parts
    parts = state.get("output_parts", [])

    # 若 output_parts 为空，尝试从 tool_results 解析
    if not parts:
        parts = _extract_parts_from_tool_results(state.get("tool_results", []))

    if not parts:
        return build_error_response(
            interface_type=interface_type,
            session_id=session_id,
            exc=RuntimeError("Workflow produced no output"),
            metadata=metadata,
        )

    # 清洗输出 parts
    cleaned_parts = _clean_output_parts(parts)

    return build_success_response(
        interface_type=interface_type,
        session_id=session_id,
        metadata=metadata,
        parts=cleaned_parts,
    )


def _extract_parts_from_tool_results(tool_results: List[str]) -> List[Dict[str, Any]]:
    """从 tool_results 中提取 parts

    解析最后一个工具返回的 JSON，提取 llm_content[0].part
    """
    if not tool_results:
        return []

    # 取最后一个结果
    last_result = tool_results[-1]

    try:
        data = json.loads(last_result) if isinstance(last_result, str) else last_result

        # 检查错误
        if data.get("error_code", 0) != 0:
            logger.warning(f"Tool result has error: {data.get('status_info')}")
            return []

        # 提取 parts
        llm_content = data.get("llm_content", [])
        if llm_content and isinstance(llm_content, list):
            return llm_content[0].get("part", [])

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.error(f"Failed to parse tool result: {e}")

    return []


def _clean_output_parts(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """清洗输出 parts，移除空值字段并规范化。

    Args:
        parts: 原始 parts 列表

    Returns:
        清洗后的 parts 列表
    """
    cleaned_parts = []
    for part in parts:
        cleaned_part = {
            "content_type": part.get("content_type"),
            "content_url": part.get("content_url"),
        }

        # 添加可选字段
        if part.get("content_text"):
            cleaned_part["content_text"] = part["content_text"]

        # 处理过期时间（统一输出为 url_expire_time）
        if part.get("url_expire_time"):
            cleaned_part["url_expire_time"] = part["url_expire_time"]
        elif part.get("expire_time"):
            cleaned_part["url_expire_time"] = part["expire_time"]

        # 移除 None 值字段
        cleaned_part = {k: v for k, v in cleaned_part.items() if v is not None}
        cleaned_parts.append(cleaned_part)

    return cleaned_parts


def extract_function_id(request_data: Dict[str, Any]) -> Optional[int]:
    """从请求中提取 function_id。

    从 llm_content[0]["part"][...]["parameter"]["function_id"] 中提取。

    Args:
        request_data: 请求数据

    Returns:
        function_id 或 None
    """
    data = ensure_dict(request_data)
    fid = extract_parameter(data, "function_id")
    if fid is not None:
        try:
            return int(fid)
        except (TypeError, ValueError):
            return None
    return None


__all__ = ["parse_request", "format_response", "extract_function_id"]
