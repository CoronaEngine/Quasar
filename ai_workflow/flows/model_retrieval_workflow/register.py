from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import (
    build_placeholder_embedding,
    get_embedding_client,
    get_recognition_db_config,
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

    from ai_modules.object_recognition.tools.vector_db import VectorDB
    from ai_modules.three_d_generate.tools.model_tools import wait_for_mesh_ready

    cfg = get_recognition_db_config()
    vector_db = VectorDB(
        db_path=cfg["db_path"],
        vector_dim=cfg["vector_dim"],
    )

    inserted_count = 0
    updated_count = 0
    failed_count = 0
    skipped_count = 0
    enriched_results: List[Dict[str, Any]] = []
    six_view_generator = get_six_view_generator()

    try:
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
            model_path = row.get("model_path", "")
            parameter = (
                row.get("parameter", {})
                if isinstance(row.get("parameter"), dict)
                else {}
            )

            if parameter.get("has_mesh_pending", False):
                wait_object_id = parameter.get("object_id") or object_id
                if six_view_generator:
                    logger.info(
                        "[Workflow][register] %s 等待后台 mesh 下载完成（用于六视图）...",
                        wait_object_id,
                    )
                    wait_for_mesh_ready(wait_object_id)
                    logger.info(
                        "[Workflow][register] %s mesh 下载已完成", wait_object_id
                    )
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
                        model_path=model_path,
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

            try:
                embedding = get_embedding_client().embed_for_storage(
                    image_paths=dedup_paths[:6],
                    text=image_prompt,
                )
            except Exception as emb_exc:  # noqa: BLE001
                logger.warning(
                    "[Workflow][register] %s 嵌入模型调用失败，降级为占位向量: %s",
                    object_id,
                    emb_exc,
                )
                embedding = build_placeholder_embedding(
                    object_id=object_id,
                    model_path=model_path,
                    vector_dim=cfg["vector_dim"],
                )

            try:
                existing = vector_db.get_object(object_id)
                if existing is None:
                    rowid = vector_db.insert_object(
                        object_id=object_id,
                        embedding=embedding,
                        name=item_name,
                        category="generated_3d",
                        image_paths=dedup_paths,
                        description=image_prompt,
                    )
                    item["register_status"] = "inserted"
                    item["register_rowid"] = rowid
                    inserted_count += 1
                else:
                    updated = vector_db.update_object(
                        object_id=object_id,
                        embedding=embedding,
                        name=item_name,
                        category="generated_3d",
                        image_paths=dedup_paths,
                        description=image_prompt,
                    )
                    if updated:
                        item["register_status"] = "updated"
                        updated_count += 1
                    else:
                        item["register_status"] = "failed"
                        item["register_error"] = "更新失败"
                        failed_count += 1
            except Exception as e:  # noqa: BLE001
                item["register_status"] = "failed"
                item["register_error"] = str(e)
                failed_count += 1

            item["object_id"] = object_id
            enriched_results.append(item)
    finally:
        vector_db.close()

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
