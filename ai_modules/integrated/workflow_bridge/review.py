from __future__ import annotations

import logging

from typing import Any, Dict, Generator, Optional, Tuple

from ....ai_workflow.bridge import RequestContext, normalize_int_function_id
from ....ai_workflow.executor import stream_workflow_from_request
from ....ai_workflow.loop_state import set_loop_global_assets

from .response import inject_function_id_to_review_stream, single_stream_response

logger = logging.getLogger(__name__)

_PROCESSED_REVIEW_BATCHES: set[str] = set()


def extract_review_submit(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = data.get("metadata", {})
    if not metadata.get("review_submit"):
        return None

    llm_content = data.get("llm_content", [])
    if not isinstance(llm_content, list):
        return None

    for entry in llm_content:
        parts = entry.get("part", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            review = (part.get("parameter") or {}).get("review")
            if isinstance(review, dict) and review.get("stage") == "submitted":
                return review

    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def normalize_review_items(
    items: Any,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    if not isinstance(items, list):
        return [], []

    all_items: list[Dict[str, Any]] = []
    active_items: list[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        normalized = dict(item)
        deleted_flag = _to_bool(
            normalized.get("is_deleted", normalized.get("deleted", False))
        )
        normalized["is_deleted"] = deleted_flag
        all_items.append(normalized)
        if not deleted_flag:
            active_items.append(normalized)

    return all_items, active_items


def build_review_resume_request(
    *,
    function_id: int,
    session_id: str,
    metadata: Dict[str, Any],
    batch_id: str,
    items: Any,
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "metadata": {
            **(metadata or {}),
            "resume_from_review": True,
            "resume_batch_id": batch_id,
        },
        "llm_content": [
            {
                "role": "user",
                "interface_type": "integrated",
                "part": [
                    {
                        "content_type": "text",
                        "content_text": "审核提交，继续执行工作流",
                        "content_url": "",
                        "parameter": {
                            "function_id": function_id,
                            "resume_from_review": True,
                            "resume_batch_id": batch_id,
                            "resume_approved_elements": items,
                        },
                    }
                ],
            }
        ],
    }


def build_state_review_resume_request(
    *,
    function_id: int,
    session_id: str,
    metadata: Dict[str, Any],
    batch_id: str,
    assets: Any,
) -> Dict[str, Any]:
    if not isinstance(assets, dict):
        assets = {}

    return {
        "session_id": session_id,
        "metadata": {
            **(metadata or {}),
            "resume_global_state_review": True,
            "resume_batch_id": batch_id,
        },
        "llm_content": [
            {
                "role": "user",
                "interface_type": "integrated",
                "part": [
                    {
                        "content_type": "text",
                        "content_text": "全局状态审核提交，继续执行主编排图",
                        "content_url": "",
                        "parameter": {
                            "function_id": function_id,
                            "resume_global_state_review": True,
                            "resume_batch_id": batch_id,
                            "resume_global_assets": assets,
                        },
                    }
                ],
            }
        ],
    }


def handle_review_submit(
    ctx: RequestContext,
) -> Optional[Generator[str, None, None]]:
    review = extract_review_submit(ctx.data)
    if review is None:
        return None

    batch_id = review.get("batch_id", "")
    function_id = review.get("function_id")
    all_items, active_items = normalize_review_items(review.get("items", []))

    if batch_id in _PROCESSED_REVIEW_BATCHES:
        logger.info("[workflow] 重复审核提交已忽略: batch_id=%s", batch_id)
        return single_stream_response(ctx.session_id, ctx.metadata, "该审核批次已处理，无需重复提交。")

    if not function_id:
        logger.error("[workflow] 审核提交缺少 function_id: batch_id=%s", batch_id)
        return single_stream_response(
            ctx.session_id,
            ctx.metadata,
            f"审核提交失败：缺少工作流 ID（batch_id={batch_id}）。",
        )

    function_id = normalize_int_function_id(function_id)
    if function_id is None:
        logger.error("[workflow] 审核提交 function_id 非法: batch_id=%s", batch_id)
        return single_stream_response(
            ctx.session_id,
            ctx.metadata,
            f"审核提交失败：工作流 ID 格式错误（batch_id={batch_id}）。",
        )

    logger.info(
        "[workflow] 收到审核提交: batch_id=%s, function_id=%s, items=%d, active=%d",
        batch_id,
        function_id,
        len(all_items),
        len(active_items),
    )
    review_type = str(review.get("review_type", "") or "").strip().lower()
    if review_type == "state_assets" or "assets" in review:
        assets = review.get("assets", {})
        if not isinstance(assets, dict):
            assets = {}
        set_loop_global_assets(ctx.session_id, assets)
        _PROCESSED_REVIEW_BATCHES.add(batch_id)
        logger.info(
            "[workflow] 全局状态审核已提交，直接写入 loop_state: batch_id=%s",
            batch_id,
        )
        return single_stream_response(
            ctx.session_id,
            ctx.metadata,
            "全局资产池已更新。",
        )

    resume_request = build_review_resume_request(
            function_id=function_id,
            session_id=ctx.session_id,
            metadata=ctx.metadata,
            batch_id=batch_id,
            items=active_items,
        )

    resumed = stream_workflow_from_request(
        resume_request,
        interface_type=ctx.interface_type,
    )
    if resumed is None:
        logger.error(
            "[workflow] 审核提交后续跑失败: batch_id=%s, function_id=%s",
            batch_id,
            function_id,
        )
        return single_stream_response(
            ctx.session_id,
            ctx.metadata,
            f"审核提交失败：无法恢复工作流（batch_id={batch_id}）。",
        )

    _PROCESSED_REVIEW_BATCHES.add(batch_id)
    return inject_function_id_to_review_stream(resumed, function_id)


__all__ = [
    "extract_review_submit",
    "normalize_review_items",
    "build_review_resume_request",
    "build_state_review_resume_request",
    "handle_review_submit",
]
