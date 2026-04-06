from __future__ import annotations

import logging
from typing import Any, Dict, List

from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import normalize_object_id
from .test_cases import get_test_case

logger = logging.getLogger(__name__)


@stream_output_node("integrated", NO_OUTPUT)
def dispatch_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """从第一步的输出中组装每个物体的检索/生成任务。"""
    metadata = state.get("metadata", {})
    global_assets = state.get("global_assets", {}) or {}

    if metadata.get("workflow_test"):
        test_case_key = metadata.get("workflow_test_case", "default")
        test_data = get_test_case(test_case_key)
        test_assets = test_data.get("global_assets", {})
        if test_assets:
            logger.info(
                "[Workflow][dispatch][TEST] 工作流测试模式，使用预定义 global_assets: "
                "test_case=%s",
                test_case_key,
            )
            global_assets = test_assets

    multi_scene = global_assets.get("multi_scene", {}) or {}

    approved = multi_scene.get("approved_elements") or state.get(
        "approved_elements",
        [],
    )
    generated_images: Dict[str, str] = multi_scene.get("generated_images") or state.get(
        "generated_images",
        {},
    )

    if not approved:
        return {"error": "无可处理的设计元素（第一步输出为空）"}

    tasks: List[Dict[str, str]] = []
    for idx, elem in enumerate(approved, start=1):
        name = elem.get("item_name", "")
        image_url = generated_images.get(name, "")
        if not image_url:
            logger.warning("[Workflow][dispatch] %s 无生成图片，跳过", name)
            continue
        object_id = normalize_object_id(name, idx)
        tasks.append(
            {
                "item_name": name,
                "object_id": object_id,
                "image_url": image_url,
                "image_prompt": elem.get("image_prompt", ""),
            }
        )

    if not tasks:
        return {"error": "所有物体均无生成图片，无法进行模型检索"}

    logger.info("[Workflow][dispatch] 组装 %s 个检索/生成任务", len(tasks))
    return {
        "intermediate": {
            **state.get("intermediate", {}),
            "retrieval_tasks": tasks,
        },
    }
