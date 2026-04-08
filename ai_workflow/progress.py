from __future__ import annotations

import queue
import threading

from typing import Any, Dict

StreamEvent = Dict[str, Any]

_STREAM_EVENT_QUEUES: dict[str, queue.Queue[StreamEvent]] = {}
_STREAM_EVENT_LOCK = threading.Lock()


def register_stream_event_queue(
    session_id: str,
) -> queue.Queue[StreamEvent]:
    """为会话注册流式事件队列。"""
    event_queue: queue.Queue[StreamEvent] = queue.Queue()
    with _STREAM_EVENT_LOCK:
        _STREAM_EVENT_QUEUES[session_id] = event_queue
    return event_queue


def unregister_stream_event_queue(
    session_id: str,
    event_queue: queue.Queue[StreamEvent] | None = None,
) -> None:
    """移除会话绑定的流式事件队列。"""
    with _STREAM_EVENT_LOCK:
        existing = _STREAM_EVENT_QUEUES.get(session_id)
        if existing is None:
            return
        if event_queue is not None and existing is not event_queue:
            return
        _STREAM_EVENT_QUEUES.pop(session_id, None)


def publish_stream_event(session_id: str, event: StreamEvent) -> bool:
    """向会话流推送一个事件；若当前不在流式上下文则忽略。"""
    with _STREAM_EVENT_LOCK:
        event_queue = _STREAM_EVENT_QUEUES.get(session_id)

    if event_queue is None:
        return False

    event_queue.put(event)
    return True


def publish_node_boundary_event(session_id: str, node_name: str) -> bool:
    """发布节点开始执行的边界事件。"""
    return publish_stream_event(
        session_id,
        {
            "kind": "boundary",
            "node_name": node_name,
            "phase": "start",
        },
    )


def publish_node_entries_event(
    session_id: str,
    node_name: str,
    entries: list[dict[str, Any]],
) -> bool:
    """发布节点内的增量 entry。"""
    return publish_stream_event(
        session_id,
        {
            "kind": "content",
            "node_name": node_name,
            "entries": entries,
        },
    )


__all__ = [
    "StreamEvent",
    "publish_node_boundary_event",
    "publish_node_entries_event",
    "publish_stream_event",
    "register_stream_event_queue",
    "unregister_stream_event_queue",
]
