from __future__ import annotations

import logging

from typing import Any, Generator, Optional

from ai_workflow.bridge import parse_request_context

from .review import handle_review_submit
from .router import (
    handle_explicit_function,
    handle_global_commands,
    handle_loop_mode,
    handle_normal_mode,
)

logger = logging.getLogger(__name__)


def resolve_and_stream_workflow(
    request_data: Any,
    *,
    interface_type: str = "integrated",
) -> Optional[Generator[str, None, None]]:
    ctx = parse_request_context(request_data, interface_type=interface_type)
    logger.debug(
        "[resolve_and_stream_workflow] text='%s', explicit_function_id=%s, session=%s",
        ctx.text,
        ctx.explicit_function_id,
        ctx.session_id,
    )

    for handler in (
        handle_review_submit,
        handle_global_commands,
        handle_explicit_function,
        handle_normal_mode,
        handle_loop_mode,
    ):
        result = handler(ctx)
        if result is not None:
            return result

    return None


__all__ = ["resolve_and_stream_workflow"]
