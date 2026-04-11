from __future__ import annotations

import logging
from typing import Any, Dict, List

from ai_tools.context import reset_current_session, set_current_session
from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import normalize_object_id
from .test_cases import get_test_case

logger = logging.getLogger(__name__)


def _retry_failed_images(
    failed_elements: List[Dict[str, str]],
    session_id: str,
) -> Dict[str, str]:
    """对上游图片生成失败的元素进行一次补偿重试。"""
    from ai_workflow.flows.integrated_multi_scene_workflow.helpers import (
        extract_image_url,
        get_generate_image_tool,
    )

    image_tool = get_generate_image_tool()
    if not image_tool:
        logger.warning("[Workflow][dispatch] 图片生成工具不可用，无法补偿重试")
        return {}

    recovered: Dict[str, str] = {}
    for elem in failed_elements:
        name = elem.get("item_name", "")
        prompt = elem.get("image_prompt", "")
        if not prompt:
            continue
        token = set_current_session(session_id)
        try:
            raw_result = image_tool.invoke({"prompt": prompt})
            image_url = extract_image_url(raw_result)
            if image_url:
                recovered[name] = image_url
                logger.info("[Workflow][dispatch] %s 补偿图片生成成功", name)
            else:
                logger.warning("[Workflow][dispatch] %s 补偿图片生成结果为空", name)
        except Exception as e:
            logger.warning("[Workflow][dispatch] %s 补偿图片生成失败: %s", name, e)
        finally:
            reset_current_session(token)

    return recovered


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
    failed_elements: List[Dict[str, str]] = []
    for idx, elem in enumerate(approved, start=1):
        name = elem.get("item_name", "")
        image_url = generated_images.get(name, "")
        if not image_url:
            failed_elements.append(elem)
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

    # 对图片生成失败的元素进行补偿重试
    if failed_elements:
        logger.info(
            "[Workflow][dispatch] %s 个元素无生成图片，尝试补偿重试",
            len(failed_elements),
        )
        session_id = str(state.get("session_id", "default") or "default")
        recovered = _retry_failed_images(failed_elements, session_id)
        for idx_offset, elem in enumerate(failed_elements, start=len(tasks) + 1):
            name = elem.get("item_name", "")
            image_url = recovered.get(name, "")
            if image_url:
                object_id = normalize_object_id(name, idx_offset)
                tasks.append(
                    {
                        "item_name": name,
                        "object_id": object_id,
                        "image_url": image_url,
                        "image_prompt": elem.get("image_prompt", ""),
                    }
                )
            else:
                logger.warning("[Workflow][dispatch] %s 补偿重试仍失败，跳过", name)

    if not tasks:
        return {"error": "所有物体均无生成图片，无法进行模型检索"}

    logger.info("[Workflow][dispatch] 组装 %s 个检索/生成任务", len(tasks))
    return {
        "intermediate": {
            **state.get("intermediate", {}),
            "retrieval_tasks": tasks,
        },
    }
