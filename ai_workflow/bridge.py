"""
工作流请求适配器 - 通用框架

负责将外部请求转换为工作流可执行的格式，支持多种 interface_type。

核心功能：
- 请求块解析（从 llm_content 中提取文本、参数等）
- 命令解析（/command 格式）
- 命令到工作流的映射
- 请求上下文构建

使用方式：
    from ai_workflow.bridge import resolve_workflow_command, parse_request_context

    # 解析命令
    ctx = parse_request_context(request_data)

    # 查询命令对应的 function_id
    function_id = resolve_workflow_command("/create_image")

    # 执行工作流
    if function_id:
        result = await run_workflow(function_id, request_data)
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ai_tools.common import ensure_dict, extract_parameter
from ai_workflow.command_registry import get_workflow_command_registry
from ai_workflow.registry import get_workflow_registry

logger = logging.getLogger(__name__)

COMMAND_PATTERN = re.compile(r"^(/\S+)(?:\s+(.*))?$")


@dataclass
class RequestContext:
    """工作流请求上下文数据类

    包含从请求数据中提取的所有必要信息。
    """

    data: Dict[str, Any]
    session_id: str
    metadata: Dict[str, Any]
    text: str
    explicit_function_id: Optional[Any]
    interface_type: str


def user_message_parts(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """返回当前用户消息的 part 列表。

    从后往前查找最后一条 role=user 的消息（与 extract_user_parts 保持一致），
    以正确处理前端携带完整对话历史的情况。

    Args:
        data: 请求数据字典

    Returns:
        part 列表
    """
    llm_content = data.get("llm_content", [])
    if not isinstance(llm_content, list) or not llm_content:
        return []

    # 从后往前找最后一条 user 消息
    for entry in reversed(llm_content):
        if not isinstance(entry, dict):
            continue
        if entry.get("role") == "user":
            parts = entry.get("part", [])
            if isinstance(parts, list):
                return [part for part in parts if isinstance(part, dict)]
            continue

    # 兜底：直接取 llm_content[0]（llm_content 无 role 字段时）
    first_entry = llm_content[0]
    if not isinstance(first_entry, dict):
        return []

    parts = first_entry.get("part", [])
    if not isinstance(parts, list):
        return []
    return [part for part in parts if isinstance(part, dict)]


def text_parts(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """返回当前用户消息中的所有文本 part。"""
    return [
        part
        for part in user_message_parts(data)
        if part.get("content_type") == "text"
    ]


def first_text_part(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """返回当前用户消息中最适合承载工作流指令/提示词的文本 part。

    优先级：
    1. 已显式绑定 function_id 的文本 part
    2. 任意 slash 指令文本 part（允许命令不在首个 text part）
    3. 第一个非空文本 part
    4. 第一个文本 part
    """
    parts = text_parts(data)
    if not parts:
        return None

    for part in parts:
        params = part.get("parameter", {})
        if isinstance(params, dict) and "function_id" in params:
            return part

    for index, part in enumerate(parts):
        text = str(part.get("content_text", "") or "").strip()
        if text and COMMAND_PATTERN.match(text):
            if index > 0:
                logger.debug(
                    "Workflow command found in non-leading text part: index=%s, text=%s",
                    index,
                    text,
                )
            return part

    for part in parts:
        if str(part.get("content_text", "") or "").strip():
            return part

    return parts[0]


def extract_text(data: Dict[str, Any]) -> str:
    """从请求数据中提取纯文本

    优先提取最可能承载工作流指令/提示词的文本 part。

    Args:
        data: 请求数据字典

    Returns:
        提取的文本，如果没有则返回空字符串
    """
    part = first_text_part(data)
    if part is None:
        return ""
    return str(part.get("content_text", "") or "").strip()


def parse_command(text: str) -> Optional[Tuple[str, str]]:
    """解析指令格式 /command [arguments]

    示例：
        "/use_workflow" -> ("/use_workflow", "")
        "/create_image 生成一只猫" -> ("/create_image", "生成一只猫")

    Args:
        text: 待解析的文本

    Returns:
        (command, argument) 元组，or None if not valid format
    """
    match = COMMAND_PATTERN.match(text.strip())
    if not match:
        return None
    command = match.group(1).lower()
    argument = (match.group(2) or "").strip()
    return command, argument


def normalize_int_function_id(value: Any) -> Optional[int]:
    """规范化 function_id 为整数

    支持 int、str 等类型转换。

    Args:
        value: 任意类型的 function_id

    Returns:
        转换后的整数，或 None 如果转换失败
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def resolve_workflow_command(command: str) -> Optional[int]:
    """查询工作流命令对应的 function_id

    优先从缓存查询，若失败则触发动态发现。

    Args:
        command: 工作流命令，如 "/create_image"

    Returns:
        对应的 function_id，或 None 如果未注册
    """
    registry = get_workflow_command_registry()
    workflow_registry = get_workflow_registry()

    function_id = registry.resolve(command)
    if function_id is not None:
        workflow_registry.discover()
        if workflow_registry.has(function_id):
            return function_id

    registry.discover()
    function_id = registry.resolve(command)
    if function_id is None:
        return None

    workflow_registry.discover()
    if workflow_registry.has(function_id):
        return function_id

    logger.warning(
        "Workflow command %s resolved to unavailable function_id=%s",
        command,
        function_id,
    )
    return None


def parse_request_context(
    request_data: Any,
    *,
    interface_type: str = "integrated",
) -> RequestContext:
    """解析请求数据为请求上下文

    从原始请求数据中提取工作流框架所需的字段。

    Args:
        request_data: 原始请求数据（dict 或 JSON 字符串）
        interface_type: 接口类型，如 "integrated"、"image" 等

    Returns:
        RequestContext 数据类实例
    """
    data = ensure_dict(request_data)
    session_id = str(data.get("session_id", "default"))
    metadata = data.get("metadata", {})
    text = extract_text(data)
    explicit_function_id = extract_parameter(data, "function_id")

    return RequestContext(
        data=data,
        session_id=session_id,
        metadata=metadata,
        text=text,
        explicit_function_id=explicit_function_id,
        interface_type=interface_type,
    )


def inject_function_id_and_prompt(
    data: Dict[str, Any],
    function_id: int,
    prompt: str,
) -> None:
    """将 function_id 和 prompt 注入到请求数据中

    修改第一个文本 part 的 content_text 和 parameter.function_id。

    Args:
        data: 请求数据字典（会被修改）
        function_id: 工作流 function_id
        prompt: 新的 prompt 文本
    """
    part = first_text_part(data)
    if part is None:
        return

    part["content_text"] = prompt
    params = part.setdefault("parameter", {})
    params["function_id"] = function_id


__all__ = [
    "RequestContext",
    "user_message_parts",
    "text_parts",
    "first_text_part",
    "extract_text",
    "parse_command",
    "normalize_int_function_id",
    "resolve_workflow_command",
    "parse_request_context",
    "inject_function_id_and_prompt",
]
