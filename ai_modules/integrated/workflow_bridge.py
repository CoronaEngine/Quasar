from __future__ import annotations

import json
import logging
import re

from typing import Any, Dict, Generator, Optional, Tuple

from ai_tools.common import build_success_response, ensure_dict, extract_parameter
from ai_workflow.command_registry import get_workflow_command_registry
from ai_workflow.executor import run_workflow_from_request, stream_workflow_from_request

from .common import build_stream_done_signal
from .loop_mode import enter_workflow_loop, exit_workflow_loop, is_workflow_loop

logger = logging.getLogger(__name__)

# 已处理的审核 batch_id 集合（幂等控制）
_PROCESSED_REVIEW_BATCHES: set[str] = set()
# 待审核上下文：batch_id -> {function_id, session_id, metadata}
_PENDING_REVIEW_CONTEXTS: dict[str, Dict[str, Any]] = {}

WORKFLOW_LOOP_ENTER_COMMAND = "/use_workflow"
WORKFLOW_LOOP_EXIT_COMMAND = "/exit_workflow"
WORKFLOW_LOOP_HELP_COMMAND = "/help"
COMMAND_PATTERN = re.compile(r"^(/\S+)(?:\s+(.*))?$")
STREAM_ONLY_WORKFLOW_MSG = "工作流仅支持流式调用，请使用 send_message_to_ai_stream。"


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


def _workflow_command_count() -> int:
    registry = get_workflow_command_registry()
    registry.discover()
    return len(registry.list_commands())


def _build_workflow_help_text() -> str:
    """生成工作流循环模式下的帮助信息，包括所有可用命令和使用方法。"""
    registry = get_workflow_command_registry()
    registry.discover()
    commands = registry.list_commands()

    if not commands:
        return (
            "当前工作流循环中暂无可用命令。\n\n"
            "退出工作流模式: /exit_workflow\n"
            "获取帮助信息: /help"
        )

    lines = [
        "📋 工作流循环模式 - 可用命令列表\n",
        "=" * 50,
        "",
    ]

    for i, (command, function_id) in enumerate(sorted(commands.items()), 1):
        lines.append(f"{i}. {command} (ID: {function_id})")

    lines.extend(
        [
            "",
            "=" * 50,
            "",
            "💡 使用方法:",
            "  /命令 你的需求",
            "",
            "📝 示例:",
            "  /create_workflow 生成一个图像处理工作流",
            "",
            "🚪 退出循环模式: /exit_workflow",
            "📖 获取帮助信息: /help",
        ]
    )

    return "\n".join(lines)


def _build_non_loop_help_text() -> str:
    """生成非循环模式下的帮助信息，提示如何进入工作流循环。"""
    command_count = _workflow_command_count()
    return (
        "🤖 欢迎使用工作流功能！\n\n"
        "当前系统已注册 {} 个工作流命令。\n\n"
        "💡 如何使用工作流:\n\n"
        "  1. 输入: /use_workflow\n"
        "  2. 然后使用 /命令 参数 来调用不同的工作流\n"
        "  3. 使用 /exit_workflow 退出循环模式\n\n"
    ).format(command_count)


def _extract_review_submit(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从请求中提取审核提交数据。

    识别条件：metadata.review_submit==true 且至少一个 part 的
    parameter.review.stage=='submitted'。
    Returns:
        审核 review 字典（含 batch_id、items 等），或 None。
    """
    metadata = data.get("metadata", {})
    if not metadata.get("review_submit"):
        return None

    llm_content = data.get("llm_content", [])
    if not isinstance(llm_content, list):
        return None

    for entry in llm_content:
        parts = entry.get("part", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            review = (part.get("parameter") or {}).get("review")
            if isinstance(review, dict) and review.get("stage") == "submitted":
                return review

    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _normalize_review_items(
    items: Any,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """标准化审核提交项并过滤软删除项。

    Returns:
        (all_items, active_items)
    """
    if not isinstance(items, list):
        return [], []

    all_items: list[Dict[str, Any]] = []
    active_items: list[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        normalized = dict(item)
        deleted_flag = _to_bool(
            normalized.get("is_deleted", normalized.get("deleted", False))
        )
        normalized["is_deleted"] = deleted_flag
        all_items.append(normalized)
        if not deleted_flag:
            active_items.append(normalized)

    return all_items, active_items


def _extract_pending_review_batch_id(response: str) -> Optional[str]:
    """从流式响应 JSON 中提取 stage=pending 的 review batch_id。"""
    try:
        data = json.loads(response)
    except Exception:
        return None

    llm_content = data.get("llm_content", [])
    if not isinstance(llm_content, list):
        return None

    for entry in llm_content:
        parts = entry.get("part", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            review = (part.get("parameter") or {}).get("review")
            if isinstance(review, dict) and review.get("stage") == "pending":
                batch_id = review.get("batch_id")
                if isinstance(batch_id, str) and batch_id:
                    return batch_id
    return None


def _stream_with_review_tracking(
    stream: Generator[str, None, None],
    *,
    function_id: Optional[int],
    session_id: str,
    metadata: Dict[str, Any],
) -> Generator[str, None, None]:
    """透传流式输出，并在出现 pending review 时记录 batch 上下文。"""
    for chunk in stream:
        batch_id = _extract_pending_review_batch_id(chunk)
        if batch_id and function_id is not None:
            _PENDING_REVIEW_CONTEXTS[batch_id] = {
                "function_id": function_id,
                "session_id": session_id,
                "metadata": metadata,
            }
            logger.info(
                "[resolve_and_stream_workflow] 记录待审核上下文: batch_id=%s, function_id=%s, session=%s",
                batch_id,
                function_id,
                session_id,
            )
        yield chunk


def _build_review_resume_request(
    *,
    function_id: int,
    session_id: str,
    metadata: Dict[str, Any],
    batch_id: str,
    items: Any,
) -> Dict[str, Any]:
    """构造审核提交后的工作流续跑请求。"""
    return {
        "session_id": session_id,
        "metadata": {
            **(metadata or {}),
            "resume_from_review": True,
            "resume_batch_id": batch_id,
        },
        "llm_content": [
            {
                "role": "user",
                "interface_type": "integrated",
                "part": [
                    {
                        "content_type": "text",
                        "content_text": "审核提交，继续执行工作流",
                        "content_url": "",
                        "parameter": {
                            "function_id": function_id,
                            "resume_from_review": True,
                            "resume_batch_id": batch_id,
                            "resume_approved_elements": items,
                        },
                    }
                ],
            }
        ],
    }


def _normalize_int_function_id(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


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

    # --- 审核提交路由 ---
    review = _extract_review_submit(data)
    if review is not None:
        batch_id = review.get("batch_id", "")
        all_items, active_items = _normalize_review_items(review.get("items", []))
        if batch_id in _PROCESSED_REVIEW_BATCHES:
            logger.info(
                f"[resolve_and_run_workflow] 重复审核提交已忽略: batch_id={batch_id}"
            )
            return _build_integrated_text_response(
                session_id, metadata, "该审核批次已处理，无需重复提交。"
            )
        _PROCESSED_REVIEW_BATCHES.add(batch_id)
        logger.info(
            f"[resolve_and_run_workflow] 收到审核提交: batch_id={batch_id}, "
            f"items={len(all_items)}, active_items={len(active_items)}"
        )
        return _build_integrated_text_response(
            session_id,
            metadata,
            f"审核已提交（batch_id={batch_id}），工作流将继续执行。",
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
        logger.info("[resolve_and_run_workflow] 非流式入口命中工作流，按策略拒绝")
        return _build_integrated_text_response(
            session_id,
            metadata,
            STREAM_ONLY_WORKFLOW_MSG,
        )

    if not is_workflow_loop(session_id):
        # 在非循环模式下处理 /help 命令
        parsed = _parse_command(text)
        if parsed is not None:
            command, _ = parsed
            if command == WORKFLOW_LOOP_HELP_COMMAND:
                logger.info("[resolve_and_run_workflow] 非循环模式：处理 /help 命令")
                return _build_integrated_text_response(
                    session_id,
                    metadata,
                    _build_non_loop_help_text(),
                )

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
            logger.info("[resolve_and_run_workflow] 非流式入口命中工作流，按策略拒绝")
            return _build_integrated_text_response(
                session_id,
                metadata,
                STREAM_ONLY_WORKFLOW_MSG,
            )
        return None

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

    if command == WORKFLOW_LOOP_HELP_COMMAND:
        logger.info("[resolve_and_run_workflow] 循环模式：处理 /help 命令")
        return _build_integrated_text_response(
            session_id,
            metadata,
            _build_workflow_help_text(),
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
    logger.info("[resolve_and_run_workflow] 循环模式命中工作流，按策略拒绝非流式执行")
    return _build_integrated_text_response(
        session_id,
        metadata,
        STREAM_ONLY_WORKFLOW_MSG,
    )


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
        "[resolve_and_stream_workflow] text='%s', explicit_function_id=%s, session=%s",
        text,
        explicit_function_id,
        session_id,
    )

    # --- 审核提交路由 ---
    review = _extract_review_submit(data)
    if review is not None:
        batch_id = review.get("batch_id", "")
        all_items, active_items = _normalize_review_items(review.get("items", []))
        if batch_id in _PROCESSED_REVIEW_BATCHES:
            logger.info(
                f"[resolve_and_stream_workflow] 重复审核提交已忽略: batch_id={batch_id}"
            )
            return _single_stream_response(
                session_id, metadata, "该审核批次已处理，无需重复提交。"
            )
        context = _PENDING_REVIEW_CONTEXTS.get(batch_id)
        if not context:
            logger.warning(
                "[resolve_and_stream_workflow] 找不到审核上下文: batch_id=%s", batch_id
            )
            return _single_stream_response(
                session_id,
                metadata,
                f"审核提交失败：未找到对应上下文（batch_id={batch_id}）。",
            )

        function_id = context.get("function_id")
        if not isinstance(function_id, int):
            logger.error(
                "[resolve_and_stream_workflow] 审核上下文 function_id 非法: batch_id=%s, function_id=%s",
                batch_id,
                function_id,
            )
            return _single_stream_response(
                session_id,
                metadata,
                f"审核提交失败：上下文缺少 function_id（batch_id={batch_id}）。",
            )

        logger.info(
            f"[resolve_and_stream_workflow] 收到审核提交: batch_id={batch_id}, "
            f"items={len(all_items)}, active_items={len(active_items)}"
        )
        resume_request = _build_review_resume_request(
            function_id=function_id,
            session_id=str(context.get("session_id") or session_id),
            metadata=context.get("metadata") or metadata,
            batch_id=batch_id,
            items=active_items,
        )
        resumed = stream_workflow_from_request(
            resume_request,
            interface_type=interface_type,
        )
        if resumed is None:
            logger.error(
                "[resolve_and_stream_workflow] 审核提交后续跑失败: batch_id=%s, function_id=%s",
                batch_id,
                function_id,
            )
            return _single_stream_response(
                session_id,
                metadata,
                f"审核提交失败：无法恢复工作流（batch_id={batch_id}）。",
            )

        _PROCESSED_REVIEW_BATCHES.add(batch_id)
        _PENDING_REVIEW_CONTEXTS.pop(batch_id, None)
        return resumed

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
        if result is None:
            return None
        tracked_function_id = _normalize_int_function_id(explicit_function_id)
        return _stream_with_review_tracking(
            result,
            function_id=tracked_function_id,
            session_id=session_id,
            metadata=metadata,
        )

    if not is_workflow_loop(session_id):
        # 在非循环模式下处理 /help 命令
        parsed = _parse_command(text)
        if parsed is not None:
            command, _ = parsed
            if command == WORKFLOW_LOOP_HELP_COMMAND:
                logger.info("[resolve_and_stream_workflow] 非循环模式：处理 /help 命令")
                return _single_stream_response(
                    session_id,
                    metadata,
                    _build_non_loop_help_text(),
                )

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
        if result is None:
            return None
        tracked_function_id = _normalize_int_function_id(
            extract_parameter(data, "function_id")
        )
        return _stream_with_review_tracking(
            result,
            function_id=tracked_function_id,
            session_id=session_id,
            metadata=metadata,
        )

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

    if command == WORKFLOW_LOOP_HELP_COMMAND:
        logger.info("[resolve_and_stream_workflow] 循环模式：处理 /help 命令")
        return _single_stream_response(
            session_id,
            metadata,
            _build_workflow_help_text(),
        )

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
    if result is None:
        return None
    return _stream_with_review_tracking(
        result,
        function_id=_normalize_int_function_id(function_id),
        session_id=session_id,
        metadata=metadata,
    )


__all__ = [
    "WORKFLOW_LOOP_ENTER_COMMAND",
    "WORKFLOW_LOOP_EXIT_COMMAND",
    "WORKFLOW_LOOP_HELP_COMMAND",
    "resolve_and_run_workflow",
    "resolve_and_stream_workflow",
]
