"""工作流流式输出装饰器。"""

from __future__ import annotations

import time

from typing import Any, Callable, Dict, List, Optional

from ai_workflow.progress import publish_node_boundary_event
from ai_workflow.state import WorkflowState

FormatterFunc = Callable[[Dict[str, Any], WorkflowState], List[Dict[str, Any]]]
NodeFunc = Callable[[WorkflowState], Dict[str, Any]]


def annotate_checkpoint_parts(
    parts: List[Dict[str, Any]],
    *,
    node_name: str,
    function_id: int | None,
) -> List[Dict[str, Any]]:
    """为 parts 注入节点级 checkpoint 元数据。"""
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
            }
        )
    return parts


def build_dialogue_entry(
    interface_type: str,
    parts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """按统一协议构建单条 dialogue entry。"""
    return {
        "role": "assistant",
        "interface_type": interface_type,
        "sent_time_stamp": int(time.time()),
        "part": parts,
    }


def build_node_dialogue_entry(
    interface_type: str,
    parts: List[Dict[str, Any]],
    *,
    node_name: str,
    function_id: int | None,
) -> Dict[str, Any]:
    """构建带节点 checkpoint 标识的 dialogue entry。"""
    annotated_parts = annotate_checkpoint_parts(
        parts,
        node_name=node_name,
        function_id=function_id,
    )
    return build_dialogue_entry(interface_type, annotated_parts)


def stream_output_node(
    interface_type: str,
    formatter: FormatterFunc,
    *,
    node_name: Optional[str] = None,
) -> Callable[[NodeFunc], NodeFunc]:
    """将纯业务节点结果转换为前端协议并自动追加到对话列表。

    装饰器行为：
    1. 若状态存在 error 或 awaiting_review，直接拦截返回空更新。
    2. 执行被装饰业务节点，业务节点只需返回纯数据字典。
    3. 调用 formatter 生成 part 列表，组装三层协议并自动加时间戳。
    4. 自动将结果追加到 dialogue_entries。
    """

    def decorator(func: NodeFunc) -> NodeFunc:
        def wrapped(state: WorkflowState) -> Dict[str, Any]:
            if state.get("error") or state.get("awaiting_review"):
                return {}

            checkpoint_node_name = node_name or func.__name__
            session_id = str(state.get("session_id", "default") or "default")
            publish_node_boundary_event(session_id, checkpoint_node_name)

            data = func(state) or {}
            if not isinstance(data, dict):
                return {"error": f"节点 {func.__name__} 返回值必须是 dict"}

            parts = formatter(data, state)
            if not parts:
                return data

            function_id = state.get("function_id")
            entry = build_node_dialogue_entry(
                interface_type,
                parts,
                node_name=checkpoint_node_name,
                function_id=function_id,
            )

            merged: Dict[str, Any] = dict(data)
            merged["dialogue_entries"] = [entry]
            return merged

        wrapped.__name__ = func.__name__
        wrapped.__doc__ = func.__doc__
        return wrapped

    return decorator


__all__ = [
    "FormatterFunc",
    "annotate_checkpoint_parts",
    "build_dialogue_entry",
    "build_node_dialogue_entry",
    "stream_output_node",
]
