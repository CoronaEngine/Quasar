from __future__ import annotations

import logging
import time
import uuid

from typing import Any, Dict, Generator, Optional

from ai_tools.common import extract_parameter
from ai_workflow.bridge import (
    RequestContext,
    inject_function_id_and_prompt,
    normalize_int_function_id,
    parse_command,
    resolve_workflow_command,
)
from ai_workflow.executor import stream_workflow_from_request
from ai_workflow.loop_state import (
    clear_loop_state,
    get_loop_global_assets,
)

from ..loop_mode import enter_workflow_loop, exit_workflow_loop, is_workflow_loop
from .response import inject_function_id_to_review_stream, single_stream_response

logger = logging.getLogger(__name__)

WORKFLOW_LOOP_ENTER_COMMAND = "/use_workflow"
WORKFLOW_LOOP_EXIT_COMMAND = "/exit_workflow"
WORKFLOW_LOOP_HELP_COMMAND = "/help"

STATE_REVIEW_COMMANDS = {"/state_review", "/state_edit", "/state_assets"}


def _available_workflow_commands() -> Dict[str, int]:
    from ai_workflow.command_registry import get_workflow_command_registry
    from ai_workflow.registry import get_workflow_registry

    command_registry = get_workflow_command_registry()
    workflow_registry = get_workflow_registry()

    command_registry.discover()
    workflow_registry.discover()

    available_function_ids = set(workflow_registry.list_function_ids())
    return {
        command: function_id
        for command, function_id in command_registry.list_commands().items()
        if function_id in available_function_ids
    }


def _workflow_command_count() -> int:
    return len(_available_workflow_commands())


def build_workflow_help_text() -> str:
    commands = _available_workflow_commands()

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
            "  /命令 [可选参数]",
            "",
            "📝 示例:",
            "  /multi_scene 生成一个客厅设计方案",
            "  /multi_scene --test 生成一个客厅设计方案",
            "  /multi_scene --test --case partial_elements 生成一个餐厅设计方案",
            "  /multi_scene --test --persist 生成一个客厅设计方案",
            "  /model_retrieval --test",
            "  某些命令可直接执行，无需额外参数",
            "",
            "🧪 工作流测试模式:",
            "  纯文本 UI 可直接在命令后附加标记：",
            "  - --test  // 启用测试模式",
            "  - --case default  // 指定测试样例名称（可选）",
            "  - --persist  // 将测试结果回写到会话状态",
            "  - --no-persist  // 显式禁止回写（默认就是 false）",
            "  结构化请求也仍支持 workflow_test / workflow_test_case / persist_to_loop_state 参数。",
            "  测试模式下工作流会使用文件内置的预定义样例数据，无需真实执行依赖步骤。",
            "",
            "🚪 退出循环模式: /exit_workflow",
            "📖 获取帮助信息: /help",
        ]
    )

    return "\n".join(lines)


def build_non_loop_help_text() -> str:
    command_count = _workflow_command_count()
    return (
        "🤖 欢迎使用工作流功能！\n\n"
        "当前系统已注册 {} 个工作流命令。\n\n"
        "💡 如何使用工作流:\n\n"
        "  1. 输入: /use_workflow\n"
        "  2. 然后使用 /命令 [可选参数] 来调用不同的工作流\n"
        "  3. 使用 /exit_workflow 退出循环模式\n\n"
    ).format(command_count)


def handle_global_commands(
    ctx: RequestContext,
) -> Optional[Generator[str, None, None]]:
    global_handlers: Dict[str, Any] = {
        WORKFLOW_LOOP_ENTER_COMMAND: _global_cmd_enter,
        WORKFLOW_LOOP_EXIT_COMMAND: _global_cmd_exit,
    }
    handler = global_handlers.get(ctx.text)
    if handler is None:
        return None
    return handler(ctx)


def _global_cmd_enter(ctx: RequestContext) -> Generator[str, None, None]:
    enter_workflow_loop(ctx.session_id)
    return single_stream_response(
        ctx.session_id,
        ctx.metadata,
        "已进入工作流模式。请使用 /命令 [可选参数] 调用工作流，使用 /exit_workflow 退出。",
    )


def _global_cmd_exit(ctx: RequestContext) -> Generator[str, None, None]:
    exit_workflow_loop(ctx.session_id)
    clear_loop_state(ctx.session_id)
    return single_stream_response(ctx.session_id, ctx.metadata, "已退出工作流模式。")


def handle_explicit_function(
    ctx: RequestContext,
) -> Optional[Generator[str, None, None]]:
    if ctx.explicit_function_id is None:
        return None

    logger.info("[workflow] 发现显式 function_id=%s", ctx.explicit_function_id)
    result = stream_workflow_from_request(ctx.data, interface_type=ctx.interface_type)
    if result is None:
        logger.warning(
            "[workflow] 流式工作流执行失败或未注册: function_id=%s",
            ctx.explicit_function_id,
        )
        return None

    logger.info("[workflow] 流式工作流执行成功，返回生成器")
    return inject_function_id_to_review_stream(
        result,
        normalize_int_function_id(ctx.explicit_function_id),
    )


def handle_normal_mode(
    ctx: RequestContext,
) -> Optional[Generator[str, None, None]]:
    if is_workflow_loop(ctx.session_id):
        return None

    parsed = parse_command(ctx.text)
    if parsed is not None and parsed[0] == WORKFLOW_LOOP_HELP_COMMAND:
        logger.info("[workflow] 非循环模式：处理 /help 命令")
        return single_stream_response(
            ctx.session_id,
            ctx.metadata,
            build_non_loop_help_text(),
        )

    # 优先尝试将斜杠命令解析为工作流（无需先进入循环模式）
    if parsed is not None:
        command, argument = parsed
        function_id = resolve_workflow_command(command)
        if function_id is not None:
            logger.info(
                "[workflow] 非循环模式命令 %s -> function_id=%s",
                command,
                function_id,
            )
            inject_function_id_and_prompt(ctx.data, function_id, argument)
            result = stream_workflow_from_request(
                ctx.data, interface_type=ctx.interface_type
            )
            if result is not None:
                return inject_function_id_to_review_stream(
                    result,
                    normalize_int_function_id(function_id),
                )

    logger.debug("[workflow] 尝试从请求中提取 function_id")
    function_id = normalize_int_function_id(extract_parameter(ctx.data, "function_id"))
    result = stream_workflow_from_request(ctx.data, interface_type=ctx.interface_type)
    if result is None:
        logger.debug("[workflow] 请求中无 function_id，工作流未命中，交给 Agent 路由")
        return None

    logger.info("[workflow] 从请求中成功执行流式工作流，返回生成器")
    return inject_function_id_to_review_stream(result, function_id)


def handle_loop_mode(
    ctx: RequestContext,
) -> Optional[Generator[str, None, None]]:
    if not is_workflow_loop(ctx.session_id):
        return None

    parsed = parse_command(ctx.text)
    if parsed is None:
        logger.debug("[workflow] 循环模式下未识别的命令格式: %s", ctx.text)
        return single_stream_response(
            ctx.session_id,
            ctx.metadata,
            "当前处于工作流模式，请使用 /命令 [可选参数]。可用命令请查看 /help。",
        )

    command, argument = parsed
    builtin_handlers: Dict[str, Any] = {
        WORKFLOW_LOOP_EXIT_COMMAND: _loop_cmd_exit,
        WORKFLOW_LOOP_HELP_COMMAND: _loop_cmd_help,
    }
    if command in builtin_handlers:
        return builtin_handlers[command](ctx, argument)

    if command in STATE_REVIEW_COMMANDS:
        return _loop_cmd_state_review(ctx)

    function_id = resolve_workflow_command(command)
    if function_id is None:
        logger.warning("[workflow] 循环模式：未识别的工作流命令: %s", command)
        return single_stream_response(
            ctx.session_id,
            ctx.metadata,
            f"未识别或当前不可用的工作流命令: {command}。请使用 /help 查看可用命令。",
        )

    inject_function_id_and_prompt(ctx.data, function_id, argument)
    logger.info(
        "[workflow] 循环模式命令 %s -> function_id=%s",
        command,
        function_id,
    )
    result = stream_workflow_from_request(ctx.data, interface_type=ctx.interface_type)
    if result is None:
        logger.error(
            "[workflow] 循环模式流式工作流执行失败: command=%s, function_id=%s",
            command,
            function_id,
        )
        return None
    return inject_function_id_to_review_stream(
        result,
        normalize_int_function_id(function_id),
    )


def _loop_cmd_exit(
    ctx: RequestContext,
    _argument: str,
) -> Generator[str, None, None]:
    exit_workflow_loop(ctx.session_id)
    clear_loop_state(ctx.session_id)
    return single_stream_response(ctx.session_id, ctx.metadata, "已退出工作流模式。")


def _loop_cmd_help(
    ctx: RequestContext,
    _argument: str,
) -> Generator[str, None, None]:
    logger.info("[workflow] 循环模式：处理 /help 命令")
    return single_stream_response(
        ctx.session_id,
        ctx.metadata,
        build_workflow_help_text(),
    )


def _loop_cmd_state_review(
    ctx: RequestContext,
) -> Generator[str, None, None]:
    """在循环模式下处理状态审核命令。"""
    logger.info("[workflow] 循环模式：处理 state_review 命令")
    assets = get_loop_global_assets(ctx.session_id)
    batch_id = str(uuid.uuid4())

    review_part = {
        "content_type": "review",
        "content_text": "请审核并编辑全局资产池，提交后将覆盖更新当前状态。",
        "content_url": "",
        "parameter": {
            "review": {
                "stage": "pending",
                "review_type": "state_assets",
                "batch_id": batch_id,
                "schema_version": 1,
                "function_id": 0,
                "assets": assets,
            },
        },
    }

    from ai_tools.common import build_success_response

    response = build_success_response(
        interface_type="integrated",
        session_id=ctx.session_id,
        metadata=ctx.metadata,
        llm_content=[
            {
                "role": "assistant",
                "interface_type": "integrated",
                "sent_time_stamp": int(time.time()),
                "part": [review_part],
            }
        ],
    )
    yield response


__all__ = [
    "WORKFLOW_LOOP_ENTER_COMMAND",
    "WORKFLOW_LOOP_EXIT_COMMAND",
    "WORKFLOW_LOOP_HELP_COMMAND",
    "build_non_loop_help_text",
    "build_workflow_help_text",
    "handle_explicit_function",
    "handle_global_commands",
    "handle_loop_mode",
    "handle_normal_mode",
]
