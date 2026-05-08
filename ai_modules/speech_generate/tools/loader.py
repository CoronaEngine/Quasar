"""
外部服务配置加载器
"""

import os
from typing import Any, Mapping

from ..configs.dataclasses import TTSConfig, AudioConfig

from ....ai_service.entrance import ai_entrance


@ai_entrance.collector.register_loader("audio")
def _load_audio_config(raw: Mapping[str, Any] | None) -> AudioConfig:
    """加载音频配置"""
    if not isinstance(raw, Mapping):
        return AudioConfig()

    return AudioConfig(
        sample_rate=int(raw.get("sample_rate", 24000)),
        bitrate=int(raw.get("bitrate", 160)),
    )


@ai_entrance.collector.register_loader("tts")
def _load_tts_config(raw: Mapping[str, Any] | None) -> TTSConfig:
    """加载 TTS 配置"""
    if not isinstance(raw, Mapping):
        return TTSConfig()

    appid = raw.get("appid")
    appid_env = raw.get("appid_env")
    if appid_env:
        appid = os.getenv(str(appid_env), appid)

    token = raw.get("token")
    token_env = raw.get("token_env")
    if token_env:
        token = os.getenv(str(token_env), token)

    return TTSConfig(
        appid=appid,
        token=token,
    )
