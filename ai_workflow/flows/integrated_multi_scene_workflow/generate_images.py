from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Dict

from ai_workflow.progress import publish_node_entries_event
from ai_workflow.state import MultiSceneWorkflowState
from ai_workflow.streaming import build_node_dialogue_entry, stream_output_node
from ai_tools.context import reset_current_session, set_current_session

from .constants import IMAGE_MAX_WORKERS
from .formatters import NO_OUTPUT, format_generate_image_progress_parts
from .helpers import extract_image_url, get_generate_image_tool
from .test_cases import get_test_case

logger = logging.getLogger(__name__)


def _publish_generate_image_progress(
    state: MultiSceneWorkflowState,
    *,
    item_name: str,
    image_url: str,
    done_count: int,
    total_count: int,
    error_message: str = "",
) -> None:
    parts = format_generate_image_progress_parts(
        item_name=item_name,
        image_url=image_url,
        done_count=done_count,
        total_count=total_count,
        error_message=error_message,
    )
    if not parts:
        return

    entry = build_node_dialogue_entry(
        "integrated",
        parts,
        node_name="generate_images",
        function_id=state.get("function_id"),
    )
    publish_node_entries_event(
        str(state.get("session_id", "default") or "default"),
        "generate_images",
        [entry],
    )


@stream_output_node("integrated", NO_OUTPUT, node_name="generate_images")
def generate_images_node(state: MultiSceneWorkflowState) -> Dict[str, Any]:
    """并发生成所有审核通过元素的图片。"""
    metadata = state.get("metadata", {})

    if metadata.get("workflow_test"):
        test_case_key = metadata.get("workflow_test_case", "default")
        test_data = get_test_case(test_case_key)
        generated_images = test_data.get("generated_images")
        if isinstance(generated_images, dict) and generated_images:
            total_count = len(generated_images)
            for index, (item_name, image_url) in enumerate(
                generated_images.items(),
                1,
            ):
                _publish_generate_image_progress(
                    state,
                    item_name=item_name,
                    image_url=image_url,
                    done_count=index,
                    total_count=total_count,
                )
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
    session_id = str(state.get("session_id", "default") or "default")

    def generate_one(element: Dict[str, str]) -> tuple[str, str, str]:
        name = element.get("item_name", "未命名")
        prompt = element.get("image_prompt", "")
        if not prompt:
            return name, "", "缺少图片生成提示词"
        token = set_current_session(session_id)
        try:
            raw_result = image_tool.invoke({"prompt": prompt})
            image_url = extract_image_url(raw_result)
            if not image_url:
                return name, "", "图片生成结果为空"
            return name, image_url, ""
        except Exception as e:
            logger.error("[Workflow][generate_images] %s 生成失败: %s", name, e)
            return name, "", str(e)
        finally:
            reset_current_session(token)

    max_workers = min(len(approved), IMAGE_MAX_WORKERS)
    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(generate_one, elem) for elem in approved]
        for future in concurrent.futures.as_completed(futures):
            try:
                name, url, error_message = future.result()
                if url:
                    generated[name] = url
                completed_count += 1
                _publish_generate_image_progress(
                    state,
                    item_name=name,
                    image_url=url,
                    done_count=completed_count,
                    total_count=len(approved),
                    error_message=error_message,
                )
            except Exception as e:
                logger.error("[Workflow][generate_images] 并发任务异常: %s", e)
                completed_count += 1
                _publish_generate_image_progress(
                    state,
                    item_name="未命名",
                    image_url="",
                    done_count=completed_count,
                    total_count=len(approved),
                    error_message=str(e),
                )

    logger.info(
        "[Workflow][generate_images] 成功生成 %s/%s 张图片",
        len(generated),
        len(approved),
    )
    return {"generated_images": generated}
