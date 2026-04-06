from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List

from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .constants import (
    GENERATION_MAX_WORKERS,
    SEARCH_DISTANCE_THRESHOLD,
    SEARCH_MAX_WORKERS,
)
from .formatters import format_retrieve_or_generate_checkpoint_parts
from .helpers import get_3d_generate_tool, get_search_tool, parse_3d_result, parse_search_result

logger = logging.getLogger(__name__)


def retrieve_single_item(task: Dict[str, Any], search_tool: Any) -> Dict[str, Any]:
    """处理单个物体检索阶段。"""
    name = task["item_name"]
    object_id = task.get("object_id", "")
    image_url = task["image_url"]
    image_prompt = task.get("image_prompt", "")

    result: Dict[str, Any] = {
        "item_name": name,
        "object_id": object_id,
        "task_index": task.get("task_index", 0),
        "input_image_url": image_url,
    }

    if not search_tool:
        result.update(
            {
                "source": "pending_generation",
                "search_status": "tool_unavailable",
            }
        )
        return result

    started_at = time.perf_counter()
    logger.info("[Workflow][retrieve] %s 开始检索", name)

    try:
        raw = search_tool.invoke(
            {
                "query_images": [image_url],
                "query_text": image_prompt,
                "top_k": 1,
            }
        )
        search_info = parse_search_result(raw)
        matches = search_info.get("matches", [])
        search_error = search_info.get("error", "")
        elapsed = time.perf_counter() - started_at

        if search_error:
            logger.warning(
                "[Workflow][retrieve] %s 检索失败，将降级生成: %s (elapsed=%.2fs)",
                name,
                search_error,
                elapsed,
            )
            result.update(
                {
                    "source": "pending_generation",
                    "search_status": "error",
                    "search_error": search_error,
                }
            )
            return result

        if matches and matches[0].get("distance", 999) < SEARCH_DISTANCE_THRESHOLD:
            best = matches[0]
            result.update(
                {
                    "source": "retrieval",
                    "object_id": best.get("object_id", ""),
                    "name": best.get("name", ""),
                    "distance": best.get("distance", 0),
                    "search_elapsed_seconds": round(elapsed, 3),
                }
            )
            logger.info(
                "[Workflow][retrieve] %s 检索命中: object_id=%s, distance=%.4f, elapsed=%.2fs",
                name,
                best.get("object_id"),
                best.get("distance", 0),
                elapsed,
            )
            return result

        best_distance = matches[0].get("distance", "N/A") if matches else "N/A"
        logger.info(
            "[Workflow][retrieve] %s 检索未命中（最佳 distance=%s, elapsed=%.2fs）",
            name,
            best_distance,
            elapsed,
        )
        result.update(
            {
                "source": "pending_generation",
                "search_status": "miss",
                "best_distance": best_distance,
                "search_elapsed_seconds": round(elapsed, 3),
            }
        )
        return result
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        logger.warning(
            "[Workflow][retrieve] %s 检索异常，将降级生成: %s (elapsed=%.2fs)",
            name,
            e,
            elapsed,
        )
        result.update(
            {
                "source": "pending_generation",
                "search_status": "error",
                "search_error": str(e),
                "search_elapsed_seconds": round(elapsed, 3),
            }
        )
        return result


def generate_single_item(task: Dict[str, Any], generate_tool: Any) -> Dict[str, Any]:
    """处理单个物体生成阶段。"""
    name = task["item_name"]
    object_id = task.get("object_id", "")
    image_url = task.get("input_image_url") or task.get("image_url", "")
    result: Dict[str, Any] = {
        "item_name": name,
        "object_id": object_id,
        "task_index": task.get("task_index", 0),
        "input_image_url": image_url,
    }

    search_error = str(task.get("search_error", "") or "").strip()

    if not generate_tool:
        error_message = "检索未命中且 3D 生成工具不可用"
        if search_error:
            error_message = f"检索失败且 3D 生成工具不可用: {search_error}"
        result.update({"source": "generation", "error": error_message})
        return result

    started_at = time.perf_counter()
    logger.info("[Workflow][generate] %s 开始 3D 生成", name)

    try:
        raw = generate_tool.invoke(
            {
                "mode": "image_to_3d",
                "images": [image_url],
                "object_id": object_id,
            }
        )
        model_info = parse_3d_result(raw)
        elapsed = time.perf_counter() - started_at

        if model_info.get("error"):
            error_message = str(model_info.get("error", "生成结果解析为空"))
            logger.error(
                "[Workflow][generate] %s 3D 生成失败: %s (elapsed=%.2fs)",
                name,
                error_message,
                elapsed,
            )
            result.update({"source": "generation", "error": error_message})
            if search_error:
                result["search_error"] = search_error
            return result

        result.update(
            {
                "source": "generation",
                "model_path": model_info.get("model_path", ""),
                "parameter": model_info.get("parameter", {}),
                "generation_elapsed_seconds": round(elapsed, 3),
            }
        )
        if search_error:
            result["search_error"] = search_error

        logger.info(
            "[Workflow][generate] %s 3D 模型生成完成: %s (elapsed=%.2fs)",
            name,
            model_info.get("model_path", ""),
            elapsed,
        )
        return result
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        logger.error(
            "[Workflow][generate] %s 3D 生成失败: %s (elapsed=%.2fs)",
            name,
            e,
            elapsed,
        )
        result.update({"source": "generation", "error": str(e)})
        if search_error:
            result["search_error"] = search_error
        return result


@stream_output_node("integrated", format_retrieve_or_generate_checkpoint_parts)
def retrieve_or_generate_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """先完成全部检索，再对未命中的物体并发生成 3D 模型。"""
    tasks = state.get("intermediate", {}).get("retrieval_tasks", [])
    if not tasks:
        return {"error": "无检索/生成任务"}

    search_tool = get_search_tool()
    if not search_tool:
        logger.warning("[Workflow][retrieve_or_generate] 检索工具不可用，将全部走生成")

    retrieval_results: List[Dict[str, Any]] = []
    pending_generation: List[Dict[str, Any]] = []

    indexed_tasks = [
        {**task, "task_index": task.get("task_index", index)}
        for index, task in enumerate(tasks, start=1)
    ]

    max_workers = min(len(indexed_tasks), SEARCH_MAX_WORKERS) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(retrieve_single_item, task, search_tool): task
            for task in indexed_tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                retrieved = future.result()
            except Exception as e:
                logger.error(
                    "[Workflow][retrieve_or_generate] %s 检索任务异常: %s",
                    task.get("item_name", "?"),
                    e,
                )
                retrieved = {
                    "item_name": task.get("item_name", "未知"),
                    "object_id": task.get("object_id", ""),
                    "task_index": task.get("task_index", 0),
                    "input_image_url": task.get("image_url", ""),
                    "source": "pending_generation",
                    "search_status": "error",
                    "search_error": str(e),
                }

            if retrieved.get("source") == "retrieval":
                retrieval_results.append(retrieved)
            else:
                pending_generation.append(retrieved)

    generated_results: List[Dict[str, Any]] = []
    if pending_generation:
        generate_tool = get_3d_generate_tool()
        if not generate_tool:
            logger.warning(
                "[Workflow][retrieve_or_generate] 3D 生成工具不可用，未命中项将返回错误"
            )

        max_workers = min(len(pending_generation), GENERATION_MAX_WORKERS) or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(generate_single_item, task, generate_tool): task
                for task in pending_generation
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    generated_results.append(future.result())
                except Exception as e:
                    logger.error(
                        "[Workflow][retrieve_or_generate] %s 生成任务异常: %s",
                        task.get("item_name", "?"),
                        e,
                    )
                    generated_results.append(
                        {
                            "item_name": task.get("item_name", "未知"),
                            "object_id": task.get("object_id", ""),
                            "task_index": task.get("task_index", 0),
                            "input_image_url": task.get("input_image_url", ""),
                            "source": "generation",
                            "error": str(e),
                        }
                    )

    results = sorted(
        retrieval_results + generated_results,
        key=lambda item: item.get("task_index", 0),
    )

    logger.info(
        "[Workflow][retrieve_or_generate] 完成: 检索命中 %s, 生成 %s, 失败 %s",
        sum(1 for row in results if row.get("source") == "retrieval"),
        sum(
            1
            for row in results
            if row.get("source") == "generation" and not row.get("error")
        ),
        sum(1 for row in results if row.get("error")),
    )

    return {"model_results": results}
