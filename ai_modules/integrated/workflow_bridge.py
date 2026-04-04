from __future__ import annotations

import logging
import re

from typing import Any, Dict, Generator, Optional, Tuple

from ai_tools.common import build_success_response, ensure_dict, extract_parameter
from ai_workflow.command_registry import get_workflow_command_registry
from ai_workflow.executor import run_workflow_from_request, stream_workflow_from_request

from .common import build_stream_done_signal
from .loop_mode import enter_workflow_loop, exit_workflow_loop, is_workflow_loop

logger = logging.getLogger(__name__)

WORKFLOW_LOOP_ENTER_COMMAND = "/use_workflow"
WORKFLOW_LOOP_EXIT_COMMAND = "/exit_workflow"
COMMAND_PATTERN = re.compile(r"^(/\S+)(?:\s+(.*))?$")


def _first_text_part(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    llm_content = data.get("llm_content", [])
    if not isinstance(llm_content, list) or not llm_content:
        return None

    parts = llm_content[0].get("part", [])
    if not isinstance(parts, list):
        return None

    for part in parts:
        if isinstance(part, dict) and part.get("content_type") == "text":
            return part
    return None


def _extract_text(data: Dict[str, Any]) -> str:
    part = _first_text_part(data)
    if part is None:
        return ""
    return str(part.get("content_text", "") or "").strip()


def _parse_command(text: str) -> Optional[Tuple[str, str]]:
    match = COMMAND_PATTERN.match(text.strip())
    if not match:
        return None
    command = match.group(1).lower()
    argument = (match.group(2) or "").strip()
    return command, argument


def _build_integrated_text_response(
    session_id: str,
    metadata: Dict[str, Any],
    text: str,
) -> str:
    return build_success_response(
        interface_type="integrated",
        session_id=session_id,
        metadata=metadata,
        llm_content=[
            {
                "role": "assistant",
                "interface_type": "integrated",
                "part": [
                    {
                        "content_type": "text",
                        "content_text": text,
                        "content_url": "",
                        "parameter": {},
                    }
                ],
            }
        ],
    )


def _single_stream_response(
    session_id: str,
    metadata: Dict[str, Any],
    text: str,
) -> Generator[str, None, None]:
    yield _build_integrated_text_response(session_id, metadata, text)
    yield build_stream_done_signal(session_id, metadata)


def _inject_function_id_and_prompt(
    data: Dict[str, Any], function_id: int, prompt: str
) -> None:
    part = _first_text_part(data)
    if part is None:
        return

    part["content_text"] = prompt
    params = part.setdefault("parameter", {})
    params["function_id"] = function_id


def _resolve_workflow_command(command: str) -> Optional[int]:
    registry = get_workflow_command_registry()
    function_id = registry.resolve(command)
    if function_id is not None:
        return function_id
    registry.discover()
    return registry.resolve(command)


def resolve_and_run_workflow(
    request_data: Any,
    *,
    interface_type: str = "integrated",
) -> Optional[str]:
    """优先处理循环模式与命令路由；未命中时返回 None 交给 Agent。"""
    data = ensure_dict(request_data)
    metadata = data.get("metadata", {})
    session_id = str(data.get("session_id", "default"))

    text = _extract_text(data)
    explicit_function_id = extract_parameter(data, "function_id")

    logger.debug(
        f"[resolve_and_run_workflow] text='{text}', explicit_function_id={explicit_function_id}, session={session_id}"
    )

    if text == WORKFLOW_LOOP_ENTER_COMMAND:
        enter_workflow_loop(session_id)
        return _build_integrated_text_response(
            session_id,
            metadata,
            "已进入工作流模式。请使用 /命令 参数 调用工作流，使用 /exit_workflow 退出。",
        )

    if text == WORKFLOW_LOOP_EXIT_COMMAND:
        exit_workflow_loop(session_id)
        return _build_integrated_text_response(
            session_id, metadata, "已退出工作流模式。"
        )

    if explicit_function_id is not None:
        logger.info(
            f"[resolve_and_run_workflow] 发现显式 function_id={explicit_function_id}"
        )
        result = run_workflow_from_request(data, interface_type=interface_type)
        if result is None:
            logger.warning(
                f"[resolve_and_run_workflow] 工作流执行失败或未注册: function_id={explicit_function_id}"
            )
        else:
            logger.info("[resolve_and_run_workflow] 工作流执行成功，返回结果")
        return result

    if not is_workflow_loop(session_id):
        logger.debug("[resolve_and_run_workflow] 尝试从请求中提取 function_id")
        result = run_workflow_from_request(
            data,
            interface_type=interface_type,
        )
        if result is None:
            logger.debug(
                "[resolve_and_run_workflow] 请求中无 function_id，工作流未命中，交给 Agent 路由"
            )
        else:
            logger.info("[resolve_and_run_workflow] 从请求中成功执行工作流，返回结果")
        return result

    parsed = _parse_command(text)
    if parsed is None:
        logger.debug(f"[resolve_and_run_workflow] 循环模式下未识别的命令格式: {text}")
        return _build_integrated_text_response(
            session_id,
            metadata,
            "当前处于工作流模式，请使用 /命令 参数。可用命令可在工作流配置中注册。",
        )

    command, argument = parsed
    if command == WORKFLOW_LOOP_EXIT_COMMAND:
        exit_workflow_loop(session_id)
        return _build_integrated_text_response(
            session_id, metadata, "已退出工作流模式。"
        )

    function_id = _resolve_workflow_command(command)
    if function_id is None:
        logger.warning(
            f"[resolve_and_run_workflow] 循环模式：未识别的工作流命令: {command}"
        )
        return _build_integrated_text_response(
            session_id,
            metadata,
            f"未识别的工作流命令: {command}。请检查 WORKFLOW_COMMANDS 注册。",
        )

    if not argument:
        logger.warning(f"[resolve_and_run_workflow] 循环模式：命令缺少参数: {command}")
        return _build_integrated_text_response(
            session_id,
            metadata,
            f"命令 {command} 缺少参数，请使用格式：{command} 你的需求",
        )

    _inject_function_id_and_prompt(data, function_id, argument)
    logger.info(
        f"[resolve_and_run_workflow] 循环模式命令 {command} -> function_id={function_id}"
    )
    result = run_workflow_from_request(
        data,
        interface_type=interface_type,
    )
    if result is None:
        logger.error(
            f"[resolve_and_run_workflow] 循环模式工作流执行失败: command={command}, function_id={function_id}"
        )
    return result


def resolve_and_stream_workflow(
    request_data: Any,
    *,
    interface_type: str = "integrated",
) -> Optional[Generator[str, None, None]]:
    """流式版本：命中工作流/循环命令时返回生成器，否则返回 None。"""
    data = ensure_dict(request_data)
    metadata = data.get("metadata", {})
    session_id = str(data.get("session_id", "default"))

    text = _extract_text(data)
    explicit_function_id = extract_parameter(data, "function_id")

    logger.debug(
        f"[resolve_and_stream_workflow] text='{text}', explicit_function_id={explicit_function_id}, session={session_id}"
    )

    if text == WORKFLOW_LOOP_ENTER_COMMAND:
        enter_workflow_loop(session_id)
        return _single_stream_response(
            session_id,
            metadata,
            "已进入工作流模式。请使用 /命令 参数 调用工作流，使用 /exit_workflow 退出。",
        )

    if text == WORKFLOW_LOOP_EXIT_COMMAND:
        exit_workflow_loop(session_id)
        return _single_stream_response(session_id, metadata, "已退出工作流模式。")

    if explicit_function_id is not None:
        logger.info(
            f"[resolve_and_stream_workflow] 发现显式 function_id={explicit_function_id}"
        )
        result = stream_workflow_from_request(data, interface_type=interface_type)
        if result is None:
            logger.warning(
                f"[resolve_and_stream_workflow] 流式工作流执行失败或未注册: function_id={explicit_function_id}"
            )
        else:
            logger.info("[resolve_and_stream_workflow] 流式工作流执行成功，返回生成器")
        return result

    if not is_workflow_loop(session_id):
        logger.debug("[resolve_and_stream_workflow] 尝试从请求中提取 function_id")
        result = stream_workflow_from_request(
            data,
            interface_type=interface_type,
        )
        if result is None:
            logger.debug(
                "[resolve_and_stream_workflow] 请求中无 function_id，工作流未命中，交给 Agent 路由"
            )
        else:
            logger.info(
                "[resolve_and_stream_workflow] 从请求中成功执行流式工作流，返回生成器"
            )
        return result

    parsed = _parse_command(text)
    if parsed is None:
        logger.debug(
            f"[resolve_and_stream_workflow] 循环模式下未识别的命令格式: {text}"
        )
        return _single_stream_response(
            session_id,
            metadata,
            "当前处于工作流模式，请使用 /命令 参数。可用命令可在工作流配置中注册。",
        )

    command, argument = parsed
    if command == WORKFLOW_LOOP_EXIT_COMMAND:
        exit_workflow_loop(session_id)
        return _single_stream_response(session_id, metadata, "已退出工作流模式。")

    function_id = _resolve_workflow_command(command)
    if function_id is None:
        logger.warning(
            f"[resolve_and_stream_workflow] 循环模式：未识别的工作流命令: {command}"
        )
        return _single_stream_response(
            session_id,
            metadata,
            f"未识别的工作流命令: {command}。请检查 WORKFLOW_COMMANDS 注册。",
        )

    if not argument:
        logger.warning(
            f"[resolve_and_stream_workflow] 循环模式：命令缺少参数: {command}"
        )
        return _single_stream_response(
            session_id,
            metadata,
            f"命令 {command} 缺少参数，请使用格式：{command} 你的需求",
        )

    _inject_function_id_and_prompt(data, function_id, argument)
    logger.info(
        f"[resolve_and_stream_workflow] 循环模式命令 {command} -> function_id={function_id}"
    )
    result = stream_workflow_from_request(
        data,
        interface_type=interface_type,
    )
    if result is None:
        logger.error(
            f"[resolve_and_stream_workflow] 循环模式流式工作流执行失败: command={command}, function_id={function_id}"
        )
    return result


__all__ = [
    "WORKFLOW_LOOP_ENTER_COMMAND",
    "WORKFLOW_LOOP_EXIT_COMMAND",
    "resolve_and_run_workflow",
    "resolve_and_stream_workflow",
]
