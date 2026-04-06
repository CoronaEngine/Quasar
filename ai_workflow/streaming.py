"""工作流流式输出装饰器。"""

from __future__ import annotations

import time

from typing import Any, Callable, Dict, List

from ai_workflow.state import WorkflowState

FormatterFunc = Callable[[Dict[str, Any], WorkflowState], List[Dict[str, Any]]]
NodeFunc = Callable[[WorkflowState], Dict[str, Any]]


def stream_output_node(
    interface_type: str,
    formatter: FormatterFunc,
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

            data = func(state) or {}
            if not isinstance(data, dict):
                return {"error": f"节点 {func.__name__} 返回值必须是 dict"}

            parts = formatter(data, state)
            if not parts:
                return data

            entry = {
                "role": "assistant",
                "interface_type": interface_type,
                "sent_time_stamp": int(time.time()),
                "part": parts,
            }

            merged: Dict[str, Any] = dict(data)
            merged["dialogue_entries"] = [entry]
            return merged

        wrapped.__name__ = func.__name__
        wrapped.__doc__ = func.__doc__
        return wrapped

    return decorator


__all__ = ["stream_output_node", "FormatterFunc"]
