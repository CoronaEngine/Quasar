"""
物体识别模块 —— 配置加载器

通过 ConfigCollector 将 settings 字典转换为 RecognitionConfig 数据类，
并注入到 AIConfig 实例上。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..configs.dataclasses import (
    EmbeddingModelConfig,
    RecognitionConfig,
    VectorDBConfig,
)
from ai_service.entrance import ai_entrance
from ai_tools.helpers import _as_bool


@ai_entrance.collector.register_loader("object_recognition")
def _load_recognition_config(raw: Mapping[str, Any] | None) -> RecognitionConfig:
    """将 settings 字典转换为 RecognitionConfig 数据类。"""
    if not isinstance(raw, Mapping):
        return RecognitionConfig()

    embedding_raw = raw.get("embedding", {})
    vector_db_raw = raw.get("vector_db", {})

    embedding_cfg = (
        EmbeddingModelConfig(**embedding_raw)
        if isinstance(embedding_raw, Mapping) and embedding_raw
        else EmbeddingModelConfig()
    )
    vector_db_cfg = (
        VectorDBConfig(**vector_db_raw)
        if isinstance(vector_db_raw, Mapping) and vector_db_raw
        else VectorDBConfig()
    )

    return RecognitionConfig(
        enable=_as_bool(raw.get("enable"), False),
        embedding=embedding_cfg,
        vector_db=vector_db_cfg,
        standard_image_count=raw.get("standard_image_count", 6),
        storage_instruction=raw.get(
            "storage_instruction",
            "Represent this document for retrieval:",
        ),
        query_instruction=raw.get(
            "query_instruction",
            "Represent the query for retrieving relevant documents:",
        ),
        auto_scan_dir=raw.get("auto_scan_dir", ""),
        auto_scan_embed=_as_bool(raw.get("auto_scan_embed"), False),
        auto_scan_max_images=raw.get("auto_scan_max_images", 6),
    )
