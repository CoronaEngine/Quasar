from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List

from ai_workflow.progress import publish_node_entries_event
from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import build_node_dialogue_entry, stream_output_node
from ai_tools.context import reset_current_session, set_current_session

from .constants import GENERATION_MAX_WORKERS
from .formatters import (
    NO_OUTPUT,
    format_generate_progress_parts,
)
from .helpers import get_3d_generate_tool, parse_3d_result
from .test_cases import get_test_case

logger = logging.getLogger(__name__)


def _publish_generate_progress(
    state: ModelRetrievalWorkflowState,
    result: Dict[str, Any],
    *,
    done_count: int,
    total_count: int,
) -> None:
    parts = format_generate_progress_parts(
        result,
        done_count=done_count,
        total_count=total_count,
    )
    if not parts:
        return

    entry = build_node_dialogue_entry(
        "integrated",
        parts,
        node_name="generate",
        function_id=state.get("function_id"),
    )
    publish_node_entries_event(
        str(state.get("session_id", "default") or "default"),
        "generate",
        [entry],
    )


def _build_mock_generate_outputs(
    state: ModelRetrievalWorkflowState,
    retrieval_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]] | None:
    """在 workflow_test 模式下根据测试样例直接构造生成阶段输出。"""
    metadata = state.get("metadata", {}) or {}
    if not metadata.get("workflow_test"):
        return None

    test_case = get_test_case(metadata.get("workflow_test_case", "default"))
    expected_results = test_case.get("expected_model_results", [])
    if not isinstance(expected_results, list) or not expected_results:
        return None

    retrieval_keys = {
        (str(item.get("item_name", "") or ""), str(item.get("object_id", "") or ""))
        for item in retrieval_results
    }

    generated_results: List[Dict[str, Any]] = []
    for expected in expected_results:
        key = (
            str(expected.get("item_name", "") or ""),
            str(expected.get("object_id", "") or ""),
        )
        if key in retrieval_keys:
            continue
        if expected.get("source") != "generation" and not expected.get("error"):
            continue
        generated_results.append(dict(expected))

    if not generated_results:
        return None

    logger.info(
        "[Workflow][generate][TEST] 使用测试样例结果: 生成 %s",
        len(generated_results),
    )
    return sorted(generated_results, key=lambda item: item.get("task_index", 0))


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
    NO_OUTPUT,
    node_name="generate",
)
def generate_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """执行生成阶段，并与检索命中结果合并。"""
    pending_generation = state.get("intermediate", {}).get("pending_generation", [])
    retrieval_results = state.get("model_results", [])

    if not isinstance(retrieval_results, list):
        retrieval_results = []

    mock_generated = _build_mock_generate_outputs(state, retrieval_results)
    if mock_generated is not None:
        total_count = len(mock_generated)
        for index, row in enumerate(mock_generated, 1):
            _publish_generate_progress(
                state,
                row,
                done_count=index,
                total_count=total_count,
            )

        results = sorted(
            retrieval_results + mock_generated,
            key=lambda item: item.get("task_index", 0),
        )
        return {
            "model_results": results,
            "intermediate": {
                **state.get("intermediate", {}),
                "pending_generation": [],
            },
        }

    if not isinstance(pending_generation, list) or not pending_generation:
        return {
            "model_results": sorted(
                retrieval_results,
                key=lambda item: item.get("task_index", 0),
            )
        }

    generate_tool = get_3d_generate_tool()
    if not generate_tool:
        logger.warning("[Workflow][generate] 3D 生成工具不可用，未命中项将返回错误")

    generated_results: List[Dict[str, Any]] = []
    completed_count = 0
    session_id = str(state.get("session_id", "default") or "default")
    max_workers = min(len(pending_generation), GENERATION_MAX_WORKERS) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(generate_single_item, task, generate_tool, session_id): task
            for task in pending_generation
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error(
                    "[Workflow][generate] %s 生成任务异常: %s",
                    task.get("item_name", "?"),
                    e,
                )
                result = {
                    "item_name": task.get("item_name", "未知"),
                    "object_id": task.get("object_id", ""),
                    "task_index": task.get("task_index", 0),
                    "input_image_url": task.get("input_image_url", ""),
                    "source": "generation",
                    "error": str(e),
                }

            generated_results.append(result)
            completed_count += 1
            _publish_generate_progress(
                state,
                result,
                done_count=completed_count,
                total_count=len(pending_generation),
            )

    results = sorted(
        retrieval_results + generated_results,
        key=lambda item: item.get("task_index", 0),
    )

    logger.info(
        "[Workflow][generate] 完成: 检索命中 %s, 生成 %s, 失败 %s",
        sum(1 for row in results if row.get("source") == "retrieval"),
        sum(
            1
            for row in results
            if row.get("source") == "generation" and not row.get("error")
        ),
        sum(1 for row in results if row.get("error")),
    )

    return {
        "model_results": results,
        "intermediate": {
            **state.get("intermediate", {}),
            "pending_generation": [],
        },
    }
