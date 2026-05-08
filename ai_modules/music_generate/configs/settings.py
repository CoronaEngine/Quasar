"""
音乐生成配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict



# ===========================================================================
# 音乐生成配置 - 默认预设
# 实际的 api_key 应放在 InnerAgentWorkflow/ai_config/media/base.py 中
# ===========================================================================


from ....ai_service.entrance import ai_entrance

@ai_entrance.collector.register_setting("music")
def MUSIC_SETTINGS() -> Dict[str, Any]:
    return {
        # Suno API 配置 (文本到背景音乐 BGM 生成)
        # 获取方式: https://www.sunoapi.org
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://api.sunoapi.org",
    }

