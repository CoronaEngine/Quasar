"""
语音合成 (TTS) 配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict


# ===========================================================================
# TTS 配置 - 默认预设
# 实际的 appid 和 token 应放在 InnerAgentWorkflow/ai_config/media/base.py 中
# ===========================================================================

from ....ai_service.entrance import ai_entrance


@ai_entrance.collector.register_setting("audio")
def AUDIO_SETTINGS() -> Dict[str, Any]:
    return {
        "sample_rate": 24000,
        "bitrate": 160,
    }


@ai_entrance.collector.register_setting("tts")
def TTS_SETTINGS() -> Dict[str, Any]:
    return {
        # 火山引擎 TTS 配置
        # 获取方式: https://www.volcengine.com/docs/6561/196768
        "appid": "YOUR_APPID_HERE",
        "token": "YOUR_TOKEN_HERE",
    }
