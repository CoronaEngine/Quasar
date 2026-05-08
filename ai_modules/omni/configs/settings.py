"""
多模态理解配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict
from ....ai_service.entrance import ai_entrance

# ===========================================================================
# 多模态理解配置 - 默认预设
# ===========================================================================


@ai_entrance.collector.register_setting("omni")
def OMNI_SETTINGS() -> Dict[str, Any]:
    return {
        "enable": True,
        "provider": "example",
        "model": "omni-model",
    }
