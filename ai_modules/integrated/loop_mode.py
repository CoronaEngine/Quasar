from __future__ import annotations

import threading


_LOOP_SESSIONS: set[str] = set()
_LOCK = threading.RLock()


def enter_workflow_loop(session_id: str) -> None:
    with _LOCK:
        _LOOP_SESSIONS.add(session_id)


def exit_workflow_loop(session_id: str) -> None:
    with _LOCK:
        _LOOP_SESSIONS.discard(session_id)


def is_workflow_loop(session_id: str) -> bool:
    with _LOCK:
        return session_id in _LOOP_SESSIONS


__all__ = ["enter_workflow_loop", "exit_workflow_loop", "is_workflow_loop"]
