# file: Backend/artificial_intelligence/service/base.py

from __future__ import annotations

import logging

from typing import Any, Dict

from ai_config.ai_config import get_ai_config
from ai_service.entrance import register_entrance
from ai_tools.common import build_error_response, ensure_dict
from ai_tools.concurrency import session_concurrency
from ai_tools.helpers import request_time_diff

from .stream_handler import handle_integrated_entrance_stream_inner

logger = logging.getLogger(__name__)


@register_entrance(handler_name="handle_integrated_entrance_stream")
def handle_integrated_entrance_stream(payload: Any):
    """
    流式统一聊天接口。
    一轮 QA 的所有 chunk 追加到同一个 entry，每次 yield 当前 entry 的最新快照。
    """
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    metadata = request_data.get("metadata", {})
    session_id = request_data.get("session_id", "default")
    cfg = get_ai_config()

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            error_response = build_error_response(
                interface_type="integrated",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
            yield error_response
            return

        # 在并发控制内执行流式处理
        yield from handle_integrated_entrance_stream_inner(
            request_data, session_id, metadata
        )


__all__ = ["handle_integrated_entrance_stream"]
