"""
物体识别模块 —— 默认设置注册

通过 ConfigCollector 将默认配置注入 ai_entrance 的全局配置树。
"""

from __future__ import annotations

from typing import Any, Dict

from ai_service.entrance import ai_entrance
from config.app_config import get_app_config


@ai_entrance.collector.register_setting("object_recognition")
def OBJECT_RECOGNITION_SETTINGS() -> Dict[str, Any]:
    """物体识别模块的默认配置"""
    db_path = str(get_app_config().paths.object_recognition_db)
    return {
        "enable": True,
        "embedding": {
            "model_size": "2B",
            "output_dim": 1024,
            "model_path": "Qwen/Qwen3-VL-Embedding-2B",
            "use_4bit": True,
            "use_flash_attention": False,
            "device_map": "auto",
        },
        "vector_db": {
            "db_path": db_path,
            "vector_dim": 1024,
            "default_top_k": 5,
        },
        "standard_image_count": 6,
        "storage_instruction": "Represent this document for retrieval:",
        "query_instruction": "Represent the query for retrieving relevant documents:",
        # 目录自动扫描
        "auto_scan_dir": "",
        "auto_scan_embed": False,
        "auto_scan_max_images": 6,
    }
