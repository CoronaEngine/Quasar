"""
图像生成配置加载器
"""

from typing import Any, Mapping

from ..configs.dataclasses import (
    ImageConstraintsConfig,
    ImageToolConfig,
)
from ....ai_service.entrance import ai_entrance
from ....ai_tools.helpers import _as_bool


@ai_entrance.collector.register_loader("image")
def _load_image_config(raw: Mapping[str, Any] | None) -> ImageToolConfig:
    """加载图像生成配置"""
    if not isinstance(raw, Mapping):
        return ImageToolConfig()

    return ImageToolConfig(
        enable=_as_bool(raw.get("enable"), False),
        provider=raw.get("provider"),
        model=raw.get("model"),
        base_url=raw.get("base_url"),
    )


@ai_entrance.collector.register_loader("image_constraints")
def _load_image_constraints_config(
    raw: Mapping[str, Any] | None,
) -> ImageConstraintsConfig:
    """加载图像约束配置"""
    if not isinstance(raw, Mapping):
        return ImageConstraintsConfig()

    return ImageConstraintsConfig(
        max_size=int(raw.get("max_size", 2000)),
        min_size=int(raw.get("min_size", 360)),
    )
