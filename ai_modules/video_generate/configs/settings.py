"""
视频生成配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict
from ....ai_service.entrance import ai_entrance


# ===========================================================================
# 视频生成配置 - 默认预设
# ===========================================================================

@ai_entrance.collector.register_setting("video")
def VIDEO_SETTINGS() -> Dict[str, Any]:
    return {
        "enable": True,
        "provider": "example",
        "model": "video-model",
    }
