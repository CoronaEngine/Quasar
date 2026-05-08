"""
图像生成配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict
from ....ai_service.entrance import ai_entrance

# ===========================================================================
# 图像生成配置 - 默认预设
# ===========================================================================


@ai_entrance.collector.register_setting("image")
def IMAGE_SETTINGS() -> Dict[str, Any]:
    return {
        "enable": True,
        "provider": "example",
        "model": "image-model",
        "base_url": "https://api.example.com/v1/images/generations",
    }


@ai_entrance.collector.register_setting("image_constraints")
def IMAGE_CONSTRAINTS_SETTINGS() -> Dict[str, Any]:
    return {
        "max_size": 2000,
        "min_size": 360,
    }
