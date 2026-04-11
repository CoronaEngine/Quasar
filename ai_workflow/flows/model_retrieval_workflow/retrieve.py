from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List

from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .constants import SEARCH_MAX_WORKERS
from .formatters import NO_OUTPUT
from .helpers import get_search_tool
from .test_cases import get_test_case

logger = logging.getLogger(__name__)


def _build_mock_retrieve_outputs(
    state: ModelRetrievalWorkflowState,
    tasks: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """在 workflow_test 模式下根据测试样例直接构造检索阶段输出。"""
    metadata = state.get("metadata", {}) or {}
    if not metadata.get("workflow_test"):
        return None

    test_case = get_test_case(metadata.get("workflow_test_case", "default"))
    expected_results = test_case.get("expected_model_results", [])
    if not isinstance(expected_results, list) or not expected_results:
        return None

    task_map: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        task_map[str(task.get("item_name", ""))] = task
        task_map[str(task.get("object_id", ""))] = task

    retrieval_results: List[Dict[str, Any]] = []
    pending_generation: List[Dict[str, Any]] = []
    for index, expected in enumerate(expected_results, start=1):
        item_name = str(expected.get("item_name", "") or "")
        object_id = str(expected.get("object_id", "") or "")
        task = task_map.get(item_name) or task_map.get(object_id) or {}

        merged: Dict[str, Any] = {
            "item_name": item_name or task.get("item_name", "未知"),
            "object_id": object_id or task.get("object_id", ""),
            "task_index": expected.get("task_index", task.get("task_index", index)),
            "input_image_url": expected.get(
                "input_image_url",
                task.get("image_url", task.get("input_image_url", "")),
            ),
            **expected,
        }

        if merged.get("source") == "retrieval" and not merged.get("error"):
            retrieval_results.append(merged)
            continue

        pending_item = dict(merged)
        pending_item["source"] = "pending_generation"
        pending_item.setdefault("search_status", "mock_pending_generation")
        pending_generation.append(pending_item)

    logger.info(
        "[Workflow][retrieve][TEST] 使用测试样例结果: 命中 %s, 待生成 %s",
        len(retrieval_results),
        len(pending_generation),
    )
    return {
        "model_results": sorted(
            retrieval_results,
            key=lambda item: item.get("task_index", 0),
        ),
        "intermediate": {
            **state.get("intermediate", {}),
            "pending_generation": sorted(
                pending_generation,
                key=lambda item: item.get("task_index", 0),
            ),
        },
    }


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
        
        # 简化：工具已返回标准化hit字段，无需手动解析
        hit = raw.get("hit", False)
        best_match = raw.get("best_match")
        error_msg = raw.get("status_info", "")
        elapsed = time.perf_counter() - started_at

        # 处理错误
        if error_msg or "error_code" in raw:
            logger.warning(
                "[Workflow][retrieve] %s 检索失败，将降级生成: %s (elapsed=%.2fs)",
                name,
                error_msg or "未知错误",
                elapsed,
            )
            result.update(
                {
                    "source": "pending_generation",
                    "search_status": "error",
                    "search_error": error_msg or "检索异常",
                }
            )
            return result

        # 处理是否命中
        if hit and best_match:
            image_paths = best_match.get("image_paths", [])
            if not isinstance(image_paths, list):
                image_paths = []
            result.update(
                {
                    "source": "retrieval",
                    "object_id": best_match.get("object_id", ""),
                    "name": best_match.get("name", ""),
                    "distance": best_match.get("distance", 0),
                    "model_path": best_match.get("model_path", ""),
                    "image_paths": image_paths,
                    "search_elapsed_seconds": round(elapsed, 3),
                }
            )
            logger.info(
                "[Workflow][retrieve] %s 检索命中: object_id=%s, distance=%.4f, elapsed=%.2fs",
                name,
                best_match.get("object_id"),
                best_match.get("distance", 0),
                elapsed,
            )
            return result

        # 未命中
        best_distance = best_match.get("distance", "N/A") if best_match else "N/A"
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


@stream_output_node(
    "integrated",
    NO_OUTPUT,
    node_name="retrieve",
)
def retrieve_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """执行检索阶段，并输出待生成任务列表。"""
    tasks = state.get("intermediate", {}).get("retrieval_tasks", [])
    if not tasks:
        return {"error": "无检索/生成任务"}

    mock_outputs = _build_mock_retrieve_outputs(state, tasks)
    if mock_outputs is not None:
        return mock_outputs

    search_tool = get_search_tool()
    if not search_tool:
        logger.warning("[Workflow][retrieve] 检索工具不可用，将全部走生成")

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
                    "[Workflow][retrieve] %s 检索任务异常: %s",
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

    logger.info(
        "[Workflow][retrieve] 完成: 检索命中 %s, 待生成 %s",
        len(retrieval_results),
        len(pending_generation),
    )

    return {
        "model_results": sorted(
            retrieval_results,
            key=lambda item: item.get("task_index", 0),
        ),
        "intermediate": {
            **state.get("intermediate", {}),
            "pending_generation": pending_generation,
        },
    }
