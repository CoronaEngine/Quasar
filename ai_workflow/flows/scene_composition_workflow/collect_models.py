"""collect_models 节点 — 从 global_assets 提取模型检索结果，构建放置列表。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT

logger = logging.getLogger(__name__)


@stream_output_node("integrated", NO_OUTPUT)
def collect_models_node(state) -> Dict[str, Any]:
    """读取上游 model_retrieval 工作流存入的模型结果，转换为放置所需的 items 列表。"""
    global_assets = state.get("global_assets", {})
    model_retrieval = global_assets.get("model_retrieval", {})
    model_results: List[Dict[str, Any]] = model_retrieval.get("model_results", [])

    if not model_results:
        return {"error": "未找到模型检索结果，请先运行 /model_retrieval 工作流"}

    placement_items: List[Dict[str, Any]] = []
    for row in model_results:
        error = row.get("error")
        if error:
            logger.warning("跳过失败模型: %s (%s)", row.get("item_name", "?"), error)
            continue

        model_path = row.get("model_path", "")
        if not model_path:
            logger.warning("跳过缺少 model_path 的模型: %s", row.get("item_name", "?"))
            continue

        item: Dict[str, Any] = {
            "object_id": row.get("object_id", row.get("item_name", "")),
            "name": row.get("item_name", ""),
            "local_path": model_path,
        }

        # 若上游提供了布局覆盖
        if row.get("pos"):
            item["pos"] = row["pos"]
        if row.get("rot"):
            item["rot"] = row["rot"]
        if row.get("scale"):
            item["scale"] = row["scale"]

        placement_items.append(item)

    if not placement_items:
        return {"error": "所有模型均失败，无法进行场景组合"}

    logger.info("collect_models: 收集到 %d 个可用模型", len(placement_items))

    return {
        "intermediate": {
            "placement_items": placement_items,
            "total_models": len(model_results),
            "valid_models": len(placement_items),
            "skipped_models": len(model_results) - len(placement_items),
        },
    }
