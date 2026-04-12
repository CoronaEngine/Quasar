"""output_result 节点 — 汇总结果写入 global_assets 并输出最终摘要。"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ai_workflow.streaming import stream_output_node

from .formatters import format_composition_result_parts

logger = logging.getLogger(__name__)


@stream_output_node("integrated", format_composition_result_parts)
def output_result_node(state) -> Dict[str, Any]:
    """汇总场景组合结果，写入 global_assets.scene_composition。"""
    intermediate = state.get("intermediate", {})

    scene_json_path = intermediate.get("scene_json_path", "")
    scene_name = intermediate.get("scene_name", "")
    imported_actors = intermediate.get("imported_actors", [])
    failed_actors = intermediate.get("failed_actors", [])
    review_result = intermediate.get("review_result", {})
    total_models = intermediate.get("total_models", 0)
    valid_models = intermediate.get("valid_models", 0)
    needs_model_regen = intermediate.get("needs_model_regen", False)

    composition_summary = {
        "scene_path": scene_json_path,
        "scene_name": scene_name,
        "total_models": total_models,
        "valid_models": valid_models,
        "imported_count": len(imported_actors),
        "failed_count": len(failed_actors),
        "imported_actors": imported_actors,
        "failed_actors": failed_actors,
        "review_result": review_result,
        "needs_model_regen": needs_model_regen,
    }

    logger.info(
        "output_result: 场景组合完成 — 导入 %d/%d, 审查 %s%s",
        len(imported_actors),
        valid_models,
        review_result.get("overall", "N/A"),
        " [需重新生成物体]" if needs_model_regen else "",
    )

    return {
        "scene_path": scene_json_path,
        "imported_actors": imported_actors,
        "failed_actors": failed_actors,
        "review_result": review_result,
        "needs_model_regen": needs_model_regen,
        "global_assets": {
            "scene_composition": composition_summary,
        },
    }
