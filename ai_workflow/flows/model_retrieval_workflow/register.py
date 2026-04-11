from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import (
    get_store_tool,
    normalize_object_id,
)
from .test_cases import get_test_case

logger = logging.getLogger(__name__)


def get_six_view_generator() -> Optional[Callable[..., List[str]]]:
    """返回六视图生成函数；当前写死为不支持。"""
    return None


@stream_output_node("integrated", NO_OUTPUT)
def register_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """将生成成功的模型写入向量数据库。"""
    model_results = state.get("model_results", [])
    if not model_results:
        return {}

    metadata = state.get("metadata", {}) or {}
    if metadata.get("workflow_test"):
        test_case = get_test_case(metadata.get("workflow_test_case", "default"))
        expected_results = test_case.get("expected_model_results", [])
        if isinstance(expected_results, list) and expected_results:
            expected_map = {
                (
                    str(item.get("item_name", "") or ""),
                    str(item.get("object_id", "") or ""),
                ): item
                for item in expected_results
                if isinstance(item, dict)
            }
            enriched_results: List[Dict[str, Any]] = []
            inserted_count = 0
            updated_count = 0
            failed_count = 0
            skipped_count = 0

            for row in model_results:
                key = (
                    str(row.get("item_name", "") or ""),
                    str(row.get("object_id", "") or ""),
                )
                merged = {**row, **expected_map.get(key, {})}
                status = str(merged.get("register_status", "") or "").lower()
                if status == "inserted":
                    inserted_count += 1
                elif status == "updated":
                    updated_count += 1
                elif status == "failed":
                    failed_count += 1
                else:
                    skipped_count += 1
                    merged.setdefault("register_status", "skipped")
                enriched_results.append(merged)

            logger.info(
                "[Workflow][register][TEST] 使用测试样例结果: inserted=%s, updated=%s, skipped=%s, failed=%s",
                inserted_count,
                updated_count,
                skipped_count,
                failed_count,
            )
            return {
                "model_results": enriched_results,
                "intermediate": {
                    **state.get("intermediate", {}),
                    "register_inserted": inserted_count,
                    "register_updated": updated_count,
                    "register_skipped": skipped_count,
                    "register_failed": failed_count,
                },
            }

    from ai_modules.three_d_generate.tools.model_tools import wait_for_mesh_ready

    store_tool = get_store_tool()
    if not store_tool:
        logger.warning("[Workflow][register] store_object 工具不可用，全部标记失败")

    inserted_count = 0
    updated_count = 0
    failed_count = 0
    skipped_count = 0
    enriched_results: List[Dict[str, Any]] = []
    six_view_generator = get_six_view_generator()

    for idx, row in enumerate(model_results, start=1):
        item = dict(row)

        if row.get("source") != "generation" or row.get("error"):
            item["register_status"] = "skipped"
            skipped_count += 1
            enriched_results.append(item)
            continue

        object_id = row.get("object_id") or normalize_object_id(
            row.get("item_name", ""),
            idx,
        )
        parameter = (
            row.get("parameter", {}) if isinstance(row.get("parameter"), dict) else {}
        )

        if parameter.get("has_mesh_pending", False):
            wait_object_id = parameter.get("object_id") or object_id
            if six_view_generator:
                logger.info(
                    "[Workflow][register] %s 等待后台 mesh 下载完成（用于六视图）...",
                    wait_object_id,
                )
                wait_for_mesh_ready(wait_object_id)
                logger.info("[Workflow][register] %s mesh 下载已完成", wait_object_id)
            else:
                logger.info(
                    "[Workflow][register] %s 六视图能力不可用，跳过后台 mesh 等待",
                    wait_object_id,
                )

        six_view_paths: List[str] = []
        if six_view_generator:
            try:
                generated = six_view_generator(
                    object_id=object_id,
                    model_path=row.get("model_path", ""),
                    parameter=parameter,
                    row=item,
                )
                if isinstance(generated, list):
                    six_view_paths = [str(path) for path in generated if path]
            except Exception as six_exc:  # noqa: BLE001
                logger.warning(
                    "[Workflow][register] %s 六视图生成失败，继续沿用原流程: %s",
                    object_id,
                    six_exc,
                )

        image_paths: List[str] = []
        preview_paths = parameter.get("preview_paths", [])
        if isinstance(preview_paths, list):
            item["preview_paths"] = [str(path) for path in preview_paths if path]
        else:
            item["preview_paths"] = []

        # 图片入嵌入优先级：六视图 > 预览图 > 输入图。
        if six_view_paths:
            image_paths.extend(six_view_paths)
        elif item["preview_paths"]:
            image_paths.extend(item["preview_paths"])
        else:
            input_image_url = row.get("input_image_url", "")
            if input_image_url:
                image_paths.append(str(input_image_url))

        dedup_paths = list(dict.fromkeys(image_paths))

        item_name = row.get("item_name", "")
        approved_elements = (
            state.get("global_assets", {})
            .get("multi_scene", {})
            .get("approved_elements", [])
        )
        image_prompt = (
            next(
                (
                    str(el.get("image_prompt", "") or "")
                    for el in approved_elements
                    if isinstance(el, dict) and el.get("item_name") == item_name
                ),
                f"{item_name or object_id or 'null'} 3D模型",
            )
            if len(dedup_paths) < 6
            else f"{item_name or object_id or 'null'} 3D模型 六视图"
        )

        if not store_tool:
            item["register_status"] = "failed"
            item["register_error"] = "store_object 工具不可用"
            failed_count += 1
            item["object_id"] = object_id
            enriched_results.append(item)
            continue

        try:
            raw_store_result = store_tool.invoke(
                {
                    "object_id": object_id,
                    "image_paths": dedup_paths,
                    "name": item_name,
                    "category": "generated_3d",
                    "description": image_prompt,
                }
            )

            # 简化：工具已返回标准化字段，无需手动解析
            register_status = raw_store_result.get("register_status", "inserted")
            rowid = raw_store_result.get("rowid")

            if register_status == "failed" or "error_code" in raw_store_result:
                # 错误处理
                error_msg = raw_store_result.get("status_info", "未知错误")
                item["register_status"] = "failed"
                item["register_error"] = error_msg
                failed_count += 1
            else:
                # 成功：直接使用返回的status
                item["register_status"] = register_status
                if rowid is not None:
                    item["register_rowid"] = rowid

                if register_status == "updated":
                    updated_count += 1
                else:
                    inserted_count += 1
        except Exception as e:  # noqa: BLE001
            item["register_status"] = "failed"
            item["register_error"] = str(e)
            failed_count += 1

        item["object_id"] = object_id
        enriched_results.append(item)

    logger.info(
        "[Workflow][register] 完成: inserted=%s, updated=%s, skipped=%s, failed=%s",
        inserted_count,
        updated_count,
        skipped_count,
        failed_count,
    )

    return {
        "model_results": enriched_results,
        "intermediate": {
            **state.get("intermediate", {}),
            "register_inserted": inserted_count,
            "register_updated": updated_count,
            "register_skipped": skipped_count,
            "register_failed": failed_count,
        },
    }
