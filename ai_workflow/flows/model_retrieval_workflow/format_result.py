from __future__ import annotations

import logging
from typing import Any, Dict

from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .formatters import format_result_checkpoint_parts

logger = logging.getLogger(__name__)


@stream_output_node("integrated", format_result_checkpoint_parts)
def format_result_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """汇总模型检索/生成结果，写入 global_assets 并输出对话内容。"""
    model_results = state.get("model_results", [])
    six_views = state.get("six_view_images", {})

    retrieval_count = sum(1 for row in model_results if row.get("source") == "retrieval")
    generation_count = sum(
        1
        for row in model_results
        if row.get("source") == "generation" and not row.get("error")
    )
    error_count = sum(1 for row in model_results if row.get("error"))

    logger.info(
        "[Workflow][format_result] 完成: 检索 %s, 生成 %s, 失败 %s, 六视图生成 %s",  # 确保这里有4个 %s
        retrieval_count,
        generation_count,
        error_count,
        len(six_views),
    )

    return {
        "global_assets": {
            "model_retrieval": {
                "model_results": model_results,
                "six_view_images": six_views,
                "retrieval_count": retrieval_count,
                "generation_count": generation_count,
                "error_count": error_count,
            }
        },
        "intermediate": {
            **state.get("intermediate", {}),
            "workflow": "model_retrieval",
            "six_view_images": six_views,
            "retrieval_count": retrieval_count,
            "generation_count": generation_count,
            "error_count": error_count,
        },
    }
