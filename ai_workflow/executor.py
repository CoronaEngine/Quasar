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
import queue
import threading
from typing import Any, Generator, Optional

from .adapter import (
    parse_request,
    format_response,
)
from .loop_state import update_loop_global_assets
from .progress import (
    publish_node_entries_event,
    publish_stream_event,
    register_stream_event_queue,
    unregister_stream_event_queue,
)
from .registry import get_workflow_registry
from .state import WorkflowState, deep_merge_dict
from ..ai_tools.common import build_error_response, build_success_response
from ..ai_tools.context import (
    set_current_session,
    reset_current_session,
)

logger = logging.getLogger(__name__)

WORKFLOW_HEARTBEAT_INTERVAL = 5.0


def _annotate_checkpoint_entries(
    entries: list[dict[str, Any]],
    *,
    node_name: str,
    function_id: int,
    checkpoint_index: int,
) -> list[dict[str, Any]]:
    """为 checkpoint 产出的 entry 注入稳定的分组标识。"""
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parts = entry.get("part", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            parameter = part.get("parameter")
            if not isinstance(parameter, dict):
                parameter = {}
                part["parameter"] = parameter
            checkpoint = parameter.get("checkpoint")
            if not isinstance(checkpoint, dict):
                checkpoint = {}
                parameter["checkpoint"] = checkpoint
            checkpoint.update(
                {
                    "entry_scope": "node",
                    "node_name": node_name,
                    "function_id": function_id,
                    "checkpoint_index": checkpoint_index,
                }
            )
    return entries


def _build_workflow_node_heartbeat(
    *,
    interface_type: str,
    session_id: str,
    metadata: dict[str, Any],
    function_id: int,
    node_name: str,
    phase: str = "start",
    boundary: bool = True,
) -> str:
    """构造工作流心跳包。"""
    return build_success_response(
        interface_type=interface_type,
        session_id=session_id,
        metadata={
            **metadata,
            "heartbeat": True,
            "workflow_node_boundary": boundary,
            "workflow_node": node_name,
            "workflow_node_phase": phase,
            "function_id": function_id,
        },
        llm_content=[],
    )


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

        # 回写 global_assets 到循环状态（工作流测试模式可控制是否回写）
        final_assets = final_state.get("global_assets")
        metadata = final_state.get("metadata", {})
        workflow_test = metadata.get("workflow_test", False)
        persist_to_loop_state = metadata.get("persist_to_loop_state", False)

        if isinstance(final_assets, dict) and final_assets:
            # 工作流测试模式：只有显式开启 persist_to_loop_state 才回写
            if workflow_test:
                if persist_to_loop_state:
                    logger.info(
                        f"[Workflow] Test mode with persist enabled: "
                        f"writing global_assets to loop_state (session={session_id})"
                    )
                    update_loop_global_assets(session_id, final_assets)
                else:
                    logger.info(
                        f"[Workflow] Test mode: skipping global_assets sync to loop_state "
                        f"(persist_to_loop_state=False, session={session_id})"
                    )
            else:
                # 正常模式：总是回写
                update_loop_global_assets(session_id, final_assets)

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
    且 state 中存在 dialogue_entries 时，yield 一次格式化的响应。

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
            # 在工作流完整执行完毕时回写 global_assets（工作流测试模式可控制是否回写）
            if not final_state.get("awaiting_review"):
                final_assets = final_state.get("global_assets")
                metadata = final_state.get("metadata", {})
                workflow_test = metadata.get("workflow_test", False)
                persist_to_loop_state = metadata.get("persist_to_loop_state", False)

                if isinstance(final_assets, dict) and final_assets:
                    if workflow_test:
                        if persist_to_loop_state:
                            logger.info(
                                f"[Stream] Test mode with persist enabled: "
                                f"writing global_assets to loop_state (session={session_id})"
                            )
                            update_loop_global_assets(session_id, final_assets)
                        else:
                            logger.info(
                                f"[Stream] Test mode: skipping global_assets sync to loop_state "
                                f"(persist_to_loop_state=False, session={session_id})"
                            )
                    else:
                        update_loop_global_assets(session_id, final_assets)
            yield format_response(final_state, interface_type=interface_type)
            return

        checkpoints = checkpoint_nodes
        next_checkpoint_index = 0
        node_checkpoint_index_map: dict[str, int] = {}

        logger.info(
            f"Streaming workflow function_id={function_id}, session={session_id}, "
            f"checkpoints={checkpoints}"
        )

        base_metadata = state.get("metadata", {})
        event_queue = register_stream_event_queue(session_id)

        def _stream_worker() -> None:
            accumulated_assets: dict[str, Any] = {}
            ended_awaiting_review = False
            worker_token = set_current_session(session_id)

            try:
                for chunk in graph.stream(state, stream_mode="updates"):
                    for node_name, node_update in chunk.items():
                        if not isinstance(node_update, dict):
                            continue

                        if "awaiting_review" in node_update:
                            ended_awaiting_review = bool(
                                node_update["awaiting_review"]
                            )

                        ga_delta = node_update.get("global_assets")
                        if isinstance(ga_delta, dict) and ga_delta:
                            accumulated_assets = deep_merge_dict(
                                accumulated_assets, ga_delta
                            )

                        if node_name not in checkpoints:
                            continue

                        new_entries = node_update.get("dialogue_entries", [])
                        if not new_entries:
                            continue

                        publish_node_entries_event(
                            session_id,
                            node_name,
                            new_entries,
                        )

                if accumulated_assets and not ended_awaiting_review:
                    metadata = state.get("metadata", {})
                    workflow_test = metadata.get("workflow_test", False)
                    persist_to_loop_state = metadata.get(
                        "persist_to_loop_state", False
                    )

                    if workflow_test:
                        if persist_to_loop_state:
                            logger.info(
                                f"[Stream] Test mode with persist enabled: "
                                f"writing accumulated_assets to loop_state (session={session_id})"
                            )
                            update_loop_global_assets(session_id, accumulated_assets)
                        else:
                            logger.info(
                                f"[Stream] Test mode: skipping accumulated_assets sync to loop_state "
                                f"(persist_to_loop_state=False, session={session_id})"
                            )
                    else:
                        update_loop_global_assets(session_id, accumulated_assets)
            except Exception as e:
                logger.error(
                    f"Streaming workflow execution failed: {e}",
                    exc_info=True,
                )
                publish_stream_event(
                    session_id,
                    {
                        "kind": "response",
                        "payload": build_error_response(
                            interface_type=interface_type,
                            session_id=session_id,
                            exc=e,
                            metadata=state.get("metadata", {}),
                        ),
                    },
                )
            finally:
                reset_current_session(worker_token)
                publish_stream_event(session_id, {"kind": "done"})

        worker = threading.Thread(
            target=_stream_worker,
            name=f"workflow-stream-{session_id}",
            daemon=True,
        )
        worker.start()

        try:
            active_node_name = ""
            while True:
                try:
                    event = event_queue.get(timeout=WORKFLOW_HEARTBEAT_INTERVAL)
                except queue.Empty:
                    yield _build_workflow_node_heartbeat(
                        interface_type=interface_type,
                        session_id=session_id,
                        metadata=base_metadata,
                        function_id=function_id,
                        node_name=active_node_name,
                        phase="keepalive",
                        boundary=False,
                    )
                    continue

                kind = event.get("kind")

                if kind == "done":
                    break

                if kind == "response":
                    payload = event.get("payload")
                    if payload:
                        yield payload
                    continue

                if kind == "boundary":
                    node_name = str(event.get("node_name", "") or "")
                    if not node_name:
                        continue
                    active_node_name = node_name
                    if node_name in checkpoints:
                        next_checkpoint_index += 1
                        node_checkpoint_index_map[node_name] = (
                            next_checkpoint_index
                        )
                    yield _build_workflow_node_heartbeat(
                        interface_type=interface_type,
                        session_id=session_id,
                        metadata=base_metadata,
                        function_id=function_id,
                        node_name=node_name,
                        phase=str(event.get("phase", "start") or "start"),
                    )
                    continue

                if kind != "content":
                    continue

                node_name = str(event.get("node_name", "") or "")
                if node_name:
                    active_node_name = node_name
                new_entries = event.get("entries", [])
                if node_name not in checkpoints or not new_entries:
                    continue

                checkpoint_index = node_checkpoint_index_map.get(node_name)
                if checkpoint_index is None:
                    next_checkpoint_index += 1
                    checkpoint_index = next_checkpoint_index
                    node_checkpoint_index_map[node_name] = checkpoint_index

                annotated_entries = _annotate_checkpoint_entries(
                    new_entries,
                    node_name=node_name,
                    function_id=function_id,
                    checkpoint_index=checkpoint_index,
                )

                response = build_success_response(
                    interface_type=interface_type,
                    session_id=session_id,
                    metadata=base_metadata,
                    llm_content=annotated_entries,
                )

                logger.info(
                    f"[Stream] Checkpoint '{node_name}': "
                    f"yield {len(new_entries)} content entries"
                )
                yield response
        finally:
            worker.join(timeout=1.0)
            unregister_stream_event_queue(session_id, event_queue)

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
    from ..ai_tools.common import (
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
    from ..ai_tools.common import (
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
