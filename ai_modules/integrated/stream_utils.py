from __future__ import annotations

import queue
import threading


_SENTINEL = object()


def run_in_thread(fn, *args, **kwargs):
    """在线程中执行函数，返回结果队列。"""
    result_q: queue.Queue = queue.Queue()

    def _worker():
        try:
            result_q.put(("ok", fn(*args, **kwargs)))
        except Exception as exc:
            result_q.put(("error", exc))

    threading.Thread(target=_worker, daemon=True).start()
    return result_q


def with_heartbeat(gen, interval: float = 5.0):
    """
    包装生成器，在没有数据产出时每隔 interval 秒 yield None。
    真实数据原样 yield。生成器结束后正常返回。
    """
    output_queue: queue.Queue = queue.Queue()

    def _producer():
        try:
            for item in gen:
                output_queue.put(item)
        except Exception as exc:
            output_queue.put(exc)
        finally:
            output_queue.put(_SENTINEL)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        try:
            item = output_queue.get(timeout=interval)
        except queue.Empty:
            yield None
            continue
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


__all__ = ["run_in_thread", "with_heartbeat"]
