"""工作流循环 session 级状态存储。

在 /use_workflow 进入工作流循环后，每次命令执行前将已有 global_assets
注入到工作流初始 state，执行完毕后将 final_state 中的 global_assets
合并回存储，实现跨工作流的状态传递。
"""

from __future__ import annotations

import threading
from typing import Any, Dict

from ai_workflow.state import deep_merge_dict

_session_assets: Dict[str, Dict[str, Any]] = {}
_lock = threading.RLock()


def get_loop_global_assets(session_id: str) -> Dict[str, Any]:
    """返回当前 session 的 global_assets 副本。"""
    with _lock:
        assets = _session_assets.get(session_id)
        if assets is None:
            return {}
        return dict(assets)


def update_loop_global_assets(session_id: str, assets: Dict[str, Any]) -> None:
    """将 assets 深度合并到当前 session 的 global_assets 中。"""
    if not isinstance(assets, dict) or not assets:
        return
    with _lock:
        existing = _session_assets.get(session_id, {})
        _session_assets[session_id] = deep_merge_dict(existing, assets)


def set_loop_global_assets(session_id: str, assets: Dict[str, Any]) -> None:
    """覆盖写入当前 session 的 global_assets（用于审核提交）。"""
    if not isinstance(assets, dict):
        assets = {}
    with _lock:
        _session_assets[session_id] = dict(assets)


def clear_loop_state(session_id: str) -> None:
    """退出循环时清除 session 状态。"""
    with _lock:
        _session_assets.pop(session_id, None)


__all__ = [
    "get_loop_global_assets",
    "update_loop_global_assets",
    "set_loop_global_assets",
    "clear_loop_state",
]
