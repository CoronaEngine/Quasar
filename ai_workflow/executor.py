"""
工作流执行器

提供统一的工作流执行入口，负责：
1. 解析请求并创建初始 State
2. 根据 function_id 获取对应的 CompiledGraph
3. 执行工作流并捕获异常
4. 格式化输出为标准响应

使用方式:
    from workflow import run_workflow

    result = run_workflow(10101, request_data)
    if result is None:
        # function_id 未注册，fallback 到原有路径
        result = handle_image_generation(request_data)
"""

from __future__ import annotations

import logging
from typing import Any, Generator, Optional

from ai_workflow.adapter import (
    parse_request,
    format_response,
)
from ai_workflow.registry import get_workflow_registry
from ai_workflow.state import WorkflowState
from ai_tools.common import build_error_response
from ai_tools.context import (
    set_current_session,
    reset_current_session,
)

logger = logging.getLogger(__name__)


def run_workflow(
    function_id: int,
    request_data: Any,
    *,
    interface_type: str = "image",
) -> Optional[str]:
    """执行指定的工作流

    根据 function_id 查找已注册的工作流，执行并返回结果。
    若 function_id 未注册，返回 None（调用方可 fallback 到原有路径）。

    Args:
        function_id: 功能 ID (如 10101, 10102, 10103)
        request_data: 原始请求数据
        interface_type: 接口类型（用于响应格式化）

    Returns:
        成功时返回标准三层 JSON 响应字符串，
        function_id 未注册时返回 None
    """
    registry = get_workflow_registry()

    # 检查是否已注册
    graph = registry.get(function_id)
    if graph is None:
        registry.discover()
        graph = registry.get(function_id)

    if graph is None:
        logger.debug(f"Workflow not registered for function_id={function_id}")
        return None

    # 解析请求
    try:
        state = parse_request(request_data)
    except Exception as e:
        logger.error(f"Failed to parse workflow request: {e}")
        return build_error_response(
            interface_type=interface_type,
            session_id=None,
            exc=e,
        )

    # 设置会话上下文
    session_id = state.get("session_id", "default")
    token = set_current_session(session_id)

    try:
        logger.info(
            f"Running workflow function_id={function_id}, session={session_id}"
        )

        # 执行工作流
        final_state: WorkflowState = graph.invoke(state)

        # 格式化输出
        return format_response(final_state, interface_type=interface_type)

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return build_error_response(
            interface_type=interface_type,
            session_id=session_id,
            exc=e,
            metadata=state.get("metadata", {}),
        )
    finally:
        reset_current_session(token)


def stream_workflow(
    function_id: int,
    request_data: Any,
    *,
    interface_type: str = "image",
    checkpoint_nodes: set[str] | None = None,
) -> Generator[str, None, None]:
    """流式执行工作流，在指定检查点节点完成时 yield 中间结果。

    使用 LangGraph 的 stream 模式逐节点执行。每当某个检查点节点完成
    且 state 中存在 output_llm_content 时，yield 一次格式化的响应。

    Args:
        function_id: 功能 ID
        request_data: 原始请求数据
        interface_type: 接口类型
        checkpoint_nodes: 需要在完成时 yield 的节点名称集合；
                          为 None 时退化为普通执行（仅在结束时 yield）

    Yields:
        标准三层 JSON 响应字符串
    """
    registry = get_workflow_registry()

    graph = registry.get(function_id)
    if graph is None:
        registry.discover()
        graph = registry.get(function_id)

    if graph is None:
        return

    try:
        state = parse_request(request_data)
    except Exception as e:
        logger.error(f"Failed to parse workflow request: {e}")
        yield build_error_response(
            interface_type=interface_type,
            session_id=None,
            exc=e,
        )
        return

    session_id = state.get("session_id", "default")
    token = set_current_session(session_id)

    try:
        # checkpoint_nodes 为 None 表示未注册检查点，退化为非流式：执行完毕后 yield 一次
        if checkpoint_nodes is None:
            logger.info(
                f"Streaming workflow (no checkpoints, single-yield) "
                f"function_id={function_id}, session={session_id}"
            )
            final_state: WorkflowState = graph.invoke(state)
            yield format_response(final_state, interface_type=interface_type)
            return

        checkpoints = checkpoint_nodes
        yielded_content_len = 0

        logger.info(
            f"Streaming workflow function_id={function_id}, session={session_id}, "
            f"checkpoints={checkpoints}"
        )

        for chunk in graph.stream(state, stream_mode="updates"):
            # chunk 格式: {"node_name": {partial_state_update}}
            for node_name, node_update in chunk.items():
                if node_name not in checkpoints:
                    continue

                # 节点更新后需要重建完整 state 来获取 output_llm_content
                # stream 模式下每个 chunk 是增量更新，需要合并
                llm_content = node_update.get("output_llm_content", [])
                if not llm_content:
                    continue

                # 仅 yield 本轮新增的 llm_content
                new_entries = llm_content[yielded_content_len:]
                if not new_entries:
                    continue

                yielded_content_len = len(llm_content)

                from ai_tools.common import build_success_response

                response = build_success_response(
                    interface_type=interface_type,
                    session_id=session_id,
                    metadata=state.get("metadata", {}),
                    llm_content=new_entries,
                )
                logger.info(
                    f"[Stream] Checkpoint '{node_name}': "
                    f"yield {len(new_entries)} content entries"
                )
                yield response

    except Exception as e:
        logger.error(f"Streaming workflow execution failed: {e}", exc_info=True)
        yield build_error_response(
            interface_type=interface_type,
            session_id=session_id,
            exc=e,
            metadata=state.get("metadata", {}),
        )
    finally:
        reset_current_session(token)


def stream_workflow_from_request(
    request_data: Any,
    *,
    interface_type: str = "image",
) -> Generator[str, None, None] | None:
    """从请求中提取 function_id 并流式执行工作流。

    返回生成器（命中工作流时）或 None（未命中时）。
    对于注册了 checkpoint_nodes 的工作流会分步输出。
    """
    from ai_tools.common import (
        ensure_dict,
        extract_parameter,
    )

    data = ensure_dict(request_data)
    function_id = extract_parameter(data, "function_id")

    if function_id is None:
        return None

    if isinstance(function_id, str):
        try:
            function_id = int(function_id)
        except ValueError:
            logger.warning(f"Invalid function_id format: {function_id}")
            return None

    # 查找该工作流是否注册了检查点节点
    checkpoint_nodes = _WORKFLOW_CHECKPOINTS.get(function_id)

    return stream_workflow(
        function_id,
        data,
        interface_type=interface_type,
        checkpoint_nodes=checkpoint_nodes,
    )


# 工作流检查点注册表：function_id → 需要在完成时 yield 中间结果的节点名集合
_WORKFLOW_CHECKPOINTS: dict[int, set[str]] = {}


def register_workflow_checkpoints(
    function_id: int, node_names: set[str]
) -> None:
    """为指定工作流注册检查点节点。"""
    _WORKFLOW_CHECKPOINTS[function_id] = node_names
    logger.debug(
        f"Registered checkpoints for function_id={function_id}: {node_names}"
    )


def run_workflow_from_request(
    request_data: Any,
    *,
    interface_type: str = "image",
) -> Optional[str]:
    """从请求中提取 function_id 并执行工作流

    便捷方法，自动从 request_data 中解析 function_id。

    Args:
        request_data: 原始请求数据
        interface_type: 接口类型

    Returns:
        成功时返回响应 JSON，未找到 function_id 或未注册时返回 None
    """
    from ai_tools.common import (
        ensure_dict,
        extract_parameter,
    )

    data = ensure_dict(request_data)
    function_id = extract_parameter(data, "function_id")

    if function_id is None:
        logger.debug("No function_id found in request")
        return None

    # 转换为 int
    if isinstance(function_id, str):
        try:
            function_id = int(function_id)
        except ValueError:
            logger.warning(f"Invalid function_id format: {function_id}")
            return None

    return run_workflow(function_id, data, interface_type=interface_type)


__all__ = [
    "run_workflow",
    "run_workflow_from_request",
    "stream_workflow",
    "stream_workflow_from_request",
    "register_workflow_checkpoints",
]
