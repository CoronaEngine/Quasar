from __future__ import annotations

import logging
from typing import Any, Dict, List

from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import (
    build_placeholder_embedding,
    get_embedding_client,
    get_recognition_db_config,
    normalize_object_id,
)

logger = logging.getLogger(__name__)


@stream_output_node("integrated", NO_OUTPUT)
def register_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """将生成成功的模型写入向量数据库。"""
    model_results = state.get("model_results", [])
    if not model_results:
        return {}

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
                logger.info("[Workflow][register] %s 等待后台 mesh 下载完成...", wait_object_id)
                wait_for_mesh_ready(wait_object_id)
                logger.info("[Workflow][register] %s mesh 下载已完成", wait_object_id)

            image_paths: List[str] = []
            preview_paths = parameter.get("preview_paths", [])
            if isinstance(preview_paths, list):
                image_paths.extend([str(path) for path in preview_paths if path])

            input_image_url = row.get("input_image_url", "")
            if input_image_url:
                image_paths.append(str(input_image_url))

            seen = set()
            dedup_paths = []
            for path in image_paths:
                if path not in seen:
                    seen.add(path)
                    dedup_paths.append(path)

            item_name = row.get("item_name", "")
            text_desc = item_name or object_id
            try:
                embedding = get_embedding_client().embed_for_storage(
                    image_paths=dedup_paths,
                    text=text_desc,
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
                        description=text_desc,
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
                        description=text_desc,
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
