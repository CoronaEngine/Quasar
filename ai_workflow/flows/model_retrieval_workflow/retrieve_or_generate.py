from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List

from ai_tools.context import reset_current_session, set_current_session
from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .constants import (
    GENERATION_MAX_WORKERS,
    SEARCH_MAX_WORKERS,
)
from .formatters import format_retrieve_or_generate_checkpoint_parts
from .helpers import get_3d_generate_tool, get_search_tool, parse_3d_result
from .test_cases import get_test_case

logger = logging.getLogger(__name__)


def _find_existing_result(
    previous_results: List[Dict[str, Any]],
    task: Dict[str, Any],
) -> Dict[str, Any] | None:
    """在已有结果中定位当前任务对应的条目，用于复核回环保留结果。"""
    task_object_id = str(task.get("object_id", "") or "")
    task_item_name = str(task.get("item_name", "") or "")

    for result in previous_results:
        result_task_object_id = str(result.get("task_object_id", "") or "")
        result_object_id = str(result.get("object_id", "") or "")
        result_item_name = str(result.get("item_name", "") or "")

        if task_object_id and task_object_id in {
            result_task_object_id,
            result_object_id,
        }:
            return result
        if task_item_name and task_item_name == result_item_name:
            return result
    return None


def _build_mock_outputs(
    state: ModelRetrievalWorkflowState,
    tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]] | None:
    """在 workflow_test 模式下直接构造合并后的检索/生成输出。"""
    metadata = state.get("metadata", {}) or {}
    if not metadata.get("workflow_test"):
        return None

    test_case = get_test_case(metadata.get("workflow_test_case", "default"))
    expected_results = test_case.get("expected_model_results", [])
    if not isinstance(expected_results, list) or not expected_results:
        return None

    task_map: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        task_map[str(task.get("item_name", "") or "")] = task
        task_map[str(task.get("object_id", "") or "")] = task

    results: List[Dict[str, Any]] = []
    for index, expected in enumerate(expected_results, start=1):
        item_name = str(expected.get("item_name", "") or "")
        object_id = str(expected.get("object_id", "") or "")
        task = task_map.get(item_name) or task_map.get(object_id) or {}
        task_object_id = str(task.get("object_id", object_id) or object_id)

        merged: Dict[str, Any] = {
            "item_name": item_name or task.get("item_name", "未知"),
            "object_id": object_id or task_object_id,
            "task_object_id": task_object_id,
            "task_index": expected.get("task_index", task.get("task_index", index)),
            "input_image_url": expected.get(
                "input_image_url",
                task.get("image_url", task.get("input_image_url", "")),
            ),
            "image_prompt": expected.get("image_prompt", task.get("image_prompt", "")),
            **expected,
        }
        results.append(merged)

    logger.info(
        "[Workflow][retrieve_or_generate][TEST] 使用测试样例结果: %s",
        len(results),
    )
    return sorted(results, key=lambda item: item.get("task_index", 0))


def retrieve_single_item(task: Dict[str, Any], search_tool: Any) -> Dict[str, Any]:
    """处理单个物体检索阶段。"""
    name = task["item_name"]
    object_id = task.get("object_id", "")
    image_url = task["image_url"]
    image_prompt = task.get("image_prompt", "")

    result: Dict[str, Any] = {
        "item_name": name,
        "object_id": object_id,
        "task_object_id": object_id,
        "task_index": task.get("task_index", 0),
        "input_image_url": image_url,
    }
    if image_prompt:
        result["image_prompt"] = image_prompt

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

        hit = raw.get("hit", False)
        best_match = raw.get("best_match")
        search_error = str(raw.get("status_info", "") or "").strip()
        elapsed = time.perf_counter() - started_at

        if search_error or "error_code" in raw:
            logger.warning(
                "[Workflow][retrieve] %s 检索失败，将降级生成: %s (elapsed=%.2fs)",
                name,
                search_error or "未知错误",
                elapsed,
            )
            result.update(
                {
                    "source": "pending_generation",
                    "search_status": "error",
                    "search_error": search_error or "检索异常",
                }
            )
            return result

        if hit and isinstance(best_match, dict):
            image_paths = best_match.get("image_paths", [])
            if not isinstance(image_paths, list):
                image_paths = []
            result.update(
                {
                    "source": "retrieval",
                    "object_id": best_match.get("object_id", "") or object_id,
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

        best_distance = (
            best_match.get("distance", "N/A")
            if isinstance(best_match, dict)
            else "N/A"
        )
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


def generate_single_item(
    task: Dict[str, Any],
    generate_tool: Any,
    session_id: str,
) -> Dict[str, Any]:
    """处理单个物体生成阶段。"""
    name = task["item_name"]
    object_id = task.get("object_id", "")
    image_url = task.get("input_image_url") or task.get("image_url", "")
    result: Dict[str, Any] = {
        "item_name": name,
        "object_id": object_id,
        "task_object_id": task.get("task_object_id", object_id),
        "task_index": task.get("task_index", 0),
        "input_image_url": image_url,
    }
    if task.get("image_prompt"):
        result["image_prompt"] = task.get("image_prompt")
    if "retry_count" in task:
        result["retry_count"] = task.get("retry_count", 0)

    search_error = str(task.get("search_error", "") or "").strip()

    if not generate_tool:
        error_message = "检索未命中且 3D 生成工具不可用"
        if search_error:
            error_message = f"检索失败且 3D 生成工具不可用: {search_error}"
        result.update({"source": "generation", "error": error_message})
        return result

    started_at = time.perf_counter()
    logger.info("[Workflow][generate] %s 开始 3D 生成", name)
    token = set_current_session(session_id)

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
    finally:
        reset_current_session(token)


@stream_output_node(
    "integrated",
    format_retrieve_or_generate_checkpoint_parts,
    node_name="retrieve_or_generate",
)
def retrieve_or_generate_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """先完成全部检索，再对未命中的物体并发生成 3D 模型。支持视觉校验的重试循环。"""
    tasks = state.get("intermediate", {}).get("retrieval_tasks", [])
    if not tasks:
        return {"error": "无检索/生成任务"}

    mock_outputs = _build_mock_outputs(state, tasks)
    if mock_outputs is not None:
        return {"model_results": mock_outputs}

    previous_results = state.get("model_results", [])
    if not isinstance(previous_results, list):
        previous_results = []

    search_tool = get_search_tool()
    if not search_tool:
        logger.warning("[Workflow][retrieve_or_generate] 检索工具不可用，将全部走生成")

    completed_results: List[Dict[str, Any]] = []
    tasks_to_retrieve: List[Dict[str, Any]] = []
    pending_generation: List[Dict[str, Any]] = []

    for index, task in enumerate(tasks, start=1):
        task_copy = {**task, "task_index": task.get("task_index", index)}
        existing_result = _find_existing_result(previous_results, task_copy)

        if existing_result:
            if existing_result.get("review_passed") or existing_result.get("source") == "retrieval":
                logger.info(
                    "[Workflow][retrieve_or_generate] %s 已合格或检索命中，直接复用结果。",
                    task_copy.get("item_name") or task_copy.get("object_id", ""),
                )
                completed_results.append(dict(existing_result))
            else:
                retry_count = int(existing_result.get("retry_count", 0) or 0)
                logger.info(
                    "[Workflow][retrieve_or_generate] %s 视觉审查不合格，进入重新生成队列 (当前重试: %s 次)。",
                    task_copy.get("item_name") or task_copy.get("object_id", ""),
                    retry_count,
                )
                task_copy["retry_count"] = retry_count
                pending_generation.append(task_copy)
        else:
            tasks_to_retrieve.append(task_copy)

    max_search_workers = min(len(tasks_to_retrieve), SEARCH_MAX_WORKERS) or 1
    if tasks_to_retrieve:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_search_workers) as pool:
            futures = {
                pool.submit(retrieve_single_item, task, search_tool): task
                for task in tasks_to_retrieve
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    retrieved = future.result()
                except Exception as e:
                    logger.error("[Workflow][retrieve] %s 异常: %s", task.get("item_name"), e)
                    retrieved = {
                        "item_name": task.get("item_name", "未知"),
                        "object_id": task.get("object_id", ""),
                        "task_object_id": task.get("object_id", ""),
                        "task_index": task.get("task_index", 0),
                        "input_image_url": task.get("image_url", ""),
                        "source": "pending_generation",
                        "search_status": "error",
                        "search_error": str(e),
                    }
                    if task.get("image_prompt"):
                        retrieved["image_prompt"] = task.get("image_prompt")

                if retrieved.get("source") == "retrieval":
                    completed_results.append(retrieved)
                else:
                    pending_generation.append(retrieved)

    generated_results: List[Dict[str, Any]] = []
    if pending_generation:
        generate_tool = get_3d_generate_tool()
        if not generate_tool:
            logger.warning("[Workflow][generate] 3D 生成工具不可用！")

        session_id = str(state.get("session_id", "default") or "default")
        max_gen_workers = min(len(pending_generation), GENERATION_MAX_WORKERS) or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_gen_workers) as pool:
            futures = {
                pool.submit(generate_single_item, task, generate_tool, session_id): task
                for task in pending_generation
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    res = future.result()
                    if "retry_count" in task:
                        res["retry_count"] = task["retry_count"]
                    generated_results.append(res)
                except Exception as e:
                    logger.error("[Workflow][generate] %s 异常: %s", task.get("item_name"), e)
                    generated_results.append({
                        "item_name": task.get("item_name", "未知"),
                        "object_id": task.get("object_id", ""),
                        "task_object_id": task.get("task_object_id", task.get("object_id", "")),
                        "task_index": task.get("task_index", 0),
                        "input_image_url": task.get("input_image_url", ""),
                        "source": "generation",
                        "error": str(e),
                    })

    results = sorted(
        completed_results + generated_results,
        key=lambda item: item.get("task_index", 0),
    )

    logger.info(
        "[Workflow][retrieve_or_generate] 轮次完成: 命中保护 %s, 生成完成 %s, 失败 %s",
        sum(1 for row in results if row.get("source") == "retrieval" or row.get("review_passed")),
        sum(1 for row in results if row.get("source") == "generation" and not row.get("error")),
        sum(1 for row in results if row.get("error")),
    )

    return {"model_results": results}
