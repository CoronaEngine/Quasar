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
import shlex
from typing import Any, Dict, List, Optional

from .bridge import first_text_part, text_parts
from .state import (
    WorkflowState,
    create_initial_state,
    deep_merge_dict,
)
from .loop_state import get_loop_global_assets
from ..ai_tools.common import (
    ensure_dict,
    extract_parameter,
    build_success_response,
    build_error_response,
)

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any) -> bool:
    """将常见字符串/数值安全转换为布尔值。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _extract_inline_workflow_test_options(prompt: str) -> tuple[str, Dict[str, Any]]:
    """从纯文本 prompt 中提取工作流测试标记。

    支持示例：
    - --test
    - --workflow-test
    - --case default
    - --case=default
    - --persist
    - --persist-to-loop-state
    """
    stripped = (prompt or "").strip()
    if not stripped:
        return "", {}

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()

    if not tokens:
        return stripped, {}

    options: Dict[str, Any] = {}
    remaining_tokens: List[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token in {"--test", "--workflow-test", "--wt"}:
            options["workflow_test"] = True
            index += 1
            continue

        if token in {"--persist", "--persist-to-loop-state"}:
            options["persist_to_loop_state"] = True
            index += 1
            continue

        if token in {"--no-persist", "--no-persist-to-loop-state"}:
            options["persist_to_loop_state"] = False
            index += 1
            continue

        if token.startswith("--case=") or token.startswith("--workflow-test-case="):
            _, value = token.split("=", 1)
            if value:
                options["workflow_test_case"] = value
            index += 1
            continue

        if token in {"--case", "--workflow-test-case"}:
            next_index = index + 1
            if next_index < len(tokens):
                options["workflow_test_case"] = tokens[next_index]
                index += 2
                continue

        remaining_tokens.append(token)
        index += 1

    return " ".join(remaining_tokens).strip(), options


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
    prompt, inline_test_options = _extract_inline_workflow_test_options(prompt)

    # 提取 images 和对应的 bounding_box（建立一一对应关系）
    images, bounding_box = _extract_images_with_bboxes(data)

    additional_type = extract_parameter(data, "additional_type", None)

    # 提取生成参数
    resolution = extract_parameter(data, "resolution", "1:1")
    image_size = _normalize_image_size(extract_parameter(data, "image_size", "2K"))

    state = create_initial_state(
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

    # 基础状态补充
    state["raw_user_input"] = prompt
    state["current_instruction"] = prompt

    # 审核提交后的续跑字段（用于跳过 analyzer/human_review）
    resume_from_review = _coerce_bool(
        extract_parameter(data, "resume_from_review", False)
    )
    resume_batch_id = extract_parameter(data, "resume_batch_id", "")
    resume_items = extract_parameter(data, "resume_approved_elements", None)

    if resume_from_review and isinstance(resume_items, list):
        state["approved_elements"] = resume_items
        state["extracted_elements"] = resume_items
        state["metadata"] = {
            **state.get("metadata", {}),
            "resume_from_review": True,
            "resume_batch_id": resume_batch_id,
        }

    # 全局状态审核提交后的续跑字段
    resume_global_state_review = _coerce_bool(
        extract_parameter(data, "resume_global_state_review", False)
    )
    resume_global_assets = extract_parameter(data, "resume_global_assets", {})
    if resume_global_state_review:
        state["metadata"] = {
            **state.get("metadata", {}),
            "resume_global_state_review": True,
            "resume_batch_id": extract_parameter(data, "resume_batch_id", ""),
        }
        if isinstance(resume_global_assets, dict):
            state["metadata"]["resume_global_assets"] = resume_global_assets

    # 从循环状态注入已积累的 global_assets
    loop_assets = get_loop_global_assets(str(session_id))
    if loop_assets:
        state["global_assets"] = deep_merge_dict(
            state.get("global_assets", {}), loop_assets
        )

    # 工作流内置测试输入模式：轻量级控制参数
    # 仅标记是否启用测试模式、使用哪个样例、以及是否回写 loop_state。
    # 具体测试内容由各工作流文件内置编码提供。
    workflow_test_param = extract_parameter(data, "workflow_test", None)
    workflow_test_case_param = extract_parameter(data, "workflow_test_case", None)
    persist_to_loop_state_param = extract_parameter(data, "persist_to_loop_state", None)

    workflow_test = (
        _coerce_bool(workflow_test_param)
        if workflow_test_param is not None
        else _coerce_bool(inline_test_options.get("workflow_test", False))
    )
    workflow_test_case = (
        workflow_test_case_param
        if workflow_test_case_param not in (None, "")
        else inline_test_options.get("workflow_test_case")
    )
    persist_to_loop_state = (
        _coerce_bool(persist_to_loop_state_param)
        if persist_to_loop_state_param is not None
        else _coerce_bool(inline_test_options.get("persist_to_loop_state", False))
    )

    if workflow_test:
        state["metadata"] = {
            **state.get("metadata", {}),
            "workflow_test": True,
            "workflow_test_case": workflow_test_case or "default",
            "persist_to_loop_state": persist_to_loop_state,
        }
        logger.info(
            f"Workflow test mode enabled: session={session_id}, "
            f"function_id={function_id}, "
            f"test_case={workflow_test_case or 'default'}, "
            f"persist_to_loop_state={persist_to_loop_state}"
        )

    return state


def _extract_prompt(data: Dict[str, Any]) -> str:
    """从 llm_content 中提取文本 prompt"""
    parts = text_parts(data)
    if not parts:
        return ""

    primary_part = first_text_part(data)
    if primary_part is not None:
        primary_params = primary_part.get("parameter", {})
        primary_text = str(primary_part.get("content_text", "") or "").strip()
        if primary_text and (
            primary_part is not parts[0]
            or isinstance(primary_params, dict)
            and any(
                key in primary_params
                for key in (
                    "function_id",
                    "resume_from_review",
                    "resume_global_state_review",
                )
            )
        ):
            return primary_text

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

    llm_content = state.get("dialogue_entries", [])
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

    # 将原始 prompt 作为文本部分返回，以便前端展示
    prompt = state.get("prompt", "")
    if prompt:
        # 检查是否已有纯文本部分，避免重复
        has_text = any(
            p.get("content_type") == "text" and p.get("content_text") == prompt
            for p in cleaned_parts
        )
        if not has_text:
            cleaned_parts.insert(
                0, {"content_type": "text", "content_text": prompt}
            )

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
