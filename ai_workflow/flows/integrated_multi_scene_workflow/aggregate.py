from __future__ import annotations

import logging
from typing import Any, Dict

from ai_workflow.state import MultiSceneWorkflowState
from ai_workflow.streaming import stream_output_node

from .formatters import format_aggregate_parts

logger = logging.getLogger(__name__)


@stream_output_node("integrated", format_aggregate_parts)
def aggregate_result_node(state: MultiSceneWorkflowState) -> Dict[str, Any]:
    """汇总物品清单、布局描述与生成图片，写入 global_assets。"""
    generated_images: Dict[str, str] = state.get("generated_images", {})
    approved = state.get("approved_elements", [])

    if not approved:
        logger.warning("[Workflow][aggregate] 无设计元素，跳过聚合")
        return {}

    logger.info(
        "[Workflow][aggregate] 完成：%s 个元素，%s 张图片成功",
        len(approved),
        len(generated_images),
    )

    return {
        "global_assets": {
            "multi_scene": {
                "approved_elements": approved,
                "generated_images": generated_images,
                "layout_text": state.get("layout_text", ""),
            }
        },
    }
