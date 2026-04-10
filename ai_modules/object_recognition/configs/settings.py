"""
物体识别模块 —— 默认设置注册

通过 ConfigCollector 将默认配置注入 ai_entrance 的全局配置树。
"""

from __future__ import annotations

from typing import Any, Dict

from ai_service.entrance import ai_entrance
from ai_config.paths_config import get_default_paths


@ai_entrance.collector.register_setting("object_recognition")
def OBJECT_RECOGNITION_SETTINGS() -> Dict[str, Any]:
    """物体识别模块的默认配置"""
    default_paths = get_default_paths()
    db_path = str(default_paths.object_recognition_db)
    return {
        "enable": True,
        "embedding": {
            "output_dim": 1024,
        },
        "vector_db": {
            "db_path": db_path,
            "vector_dim": 1024,
            "default_top_k": 5,
        },
        "standard_image_count": 6,
        "storage_instruction": "Represent this document for retrieval:",
        "query_instruction": "Represent the query for retrieving relevant documents:",
        # 云端嵌入服务（Dashscope）
        "dashscope_api_key": "",
        "dashscope_model": "tongyi-embedding-vision-plus-2026-03-06",
        # 目录自动扫描
        "auto_scan_dir": str(default_paths.assets_model_dir),
        "auto_scan_embed": True,
        "auto_scan_max_images": 6,
    }
