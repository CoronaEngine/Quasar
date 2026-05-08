"""
外部服务配置加载器
"""

import os
from typing import Any, Mapping

from ..configs.dataclasses import (
    MusicConfig,
)

from ....ai_service.entrance import ai_entrance


@ai_entrance.collector.register_loader('music')
def _load_music_config(raw: Mapping[str, Any] | None) -> MusicConfig:
    """加载音乐生成配置"""
    if not isinstance(raw, Mapping):
        return MusicConfig()

    api_key = raw.get("api_key")
    api_key_env = raw.get("api_key_env")
    if api_key_env:
        api_key = os.getenv(str(api_key_env), api_key)

    base_url = raw.get("base_url")

    return MusicConfig(
        api_key=api_key,
        base_url=base_url,
    )
