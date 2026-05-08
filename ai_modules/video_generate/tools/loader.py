"""
视频生成配置加载器
"""

from typing import Any, Mapping

from ..configs.dataclasses import VideoToolConfig
from ....ai_service.entrance import ai_entrance
from ....ai_tools.helpers import _as_bool


@ai_entrance.collector.register_loader("video")
def _load_video_config(raw: Mapping[str, Any] | None) -> VideoToolConfig:
    """加载视频生成配置"""
    if not isinstance(raw, Mapping):
        return VideoToolConfig()

    return VideoToolConfig(
        enable=_as_bool(raw.get("enable"), False),
        provider=raw.get("provider"),
        model=raw.get("model"),
        base_url=raw.get("base_url"),
    )
