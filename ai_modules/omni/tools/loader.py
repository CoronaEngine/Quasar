"""
多模态理解配置加载器
"""

from typing import Any, Mapping

from ..configs.dataclasses import OmniModelConfig
from ....ai_service.entrance import ai_entrance
from ....ai_tools.helpers import _as_bool


@ai_entrance.collector.register_loader("omni")
def _load_omni_config(raw: Mapping[str, Any] | None) -> OmniModelConfig:
    """加载多模态理解配置"""
    if not isinstance(raw, Mapping):
        return OmniModelConfig()

    return OmniModelConfig(
        enable=_as_bool(raw.get("enable"), False),
        provider=raw.get("provider"),
        model=raw.get("model"),
        max_frames=int(raw.get("max_frames", 16)),
        fps=float(raw.get("fps", 1.0)),
        image_detail=str(raw.get("image_detail", "high")),
        request_timeout=float(raw.get("request_timeout", 150.0)),
    )
