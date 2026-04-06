from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Dict

from ai_workflow.state import MultiSceneWorkflowState
from ai_workflow.streaming import stream_output_node

from .constants import IMAGE_MAX_WORKERS
from .formatters import NO_OUTPUT
from .helpers import extract_image_url, get_generate_image_tool
from .test_cases import get_test_case

logger = logging.getLogger(__name__)


@stream_output_node("integrated", NO_OUTPUT)
def generate_images_node(state: MultiSceneWorkflowState) -> Dict[str, Any]:
    """并发生成所有审核通过元素的图片。"""
    metadata = state.get("metadata", {})

    if metadata.get("workflow_test"):
        test_case_key = metadata.get("workflow_test_case", "default")
        test_data = get_test_case(test_case_key)
        generated_images = test_data.get("generated_images")
        if isinstance(generated_images, dict) and generated_images:
            logger.info(
                "[Workflow][generate_images][TEST] 工作流测试模式，使用预定义 generated_images: "
                "test_case=%s, count=%s",
                test_case_key,
                len(generated_images),
            )
            return {"generated_images": generated_images}

    approved = state.get("approved_elements", [])
    if not approved:
        logger.warning("[Workflow][generate_images] 无审核通过的元素")
        return {"generated_images": {}}

    image_tool = get_generate_image_tool()
    if not image_tool:
        logger.warning("[Workflow][generate_images] 图片生成工具不可用")
        return {"generated_images": {}}

    generated: Dict[str, str] = {}

    def generate_one(element: Dict[str, str]) -> tuple[str, str]:
        name = element.get("item_name", "未命名")
        prompt = element.get("image_prompt", "")
        if not prompt:
            return name, ""
        try:
            raw_result = image_tool.invoke({"prompt": prompt})
            image_url = extract_image_url(raw_result)
            return name, image_url
        except Exception as e:
            logger.error("[Workflow][generate_images] %s 生成失败: %s", name, e)
            return name, ""

    max_workers = min(len(approved), IMAGE_MAX_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(generate_one, elem) for elem in approved]
        for future in concurrent.futures.as_completed(futures):
            try:
                name, url = future.result()
                if url:
                    generated[name] = url
            except Exception as e:
                logger.error("[Workflow][generate_images] 并发任务异常: %s", e)

    logger.info(
        "[Workflow][generate_images] 成功生成 %s/%s 张图片",
        len(generated),
        len(approved),
    )
    return {"generated_images": generated}
