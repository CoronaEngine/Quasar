"""
网络配置加载器
"""

from typing import Any, Mapping
from ..configs.dataclasses import PollingConfig
from ....ai_service.entrance import ai_entrance


@ai_entrance.collector.register_loader('polling')
def _load_polling_config(raw: Mapping[str, Any] | None) -> PollingConfig:
    """加载轮询配置"""
    if not isinstance(raw, Mapping):
        return PollingConfig()

    service_intervals = raw.get("service_intervals", {})
    if not isinstance(service_intervals, Mapping):
        service_intervals = {}

    return PollingConfig(
        max_wait_seconds=int(raw.get("max_wait_seconds", 150)),
        default_interval=float(raw.get("default_interval", 3.0)),
        speech_interval=float(service_intervals.get("speech", 2.0)),
        music_interval=float(service_intervals.get("music", 5.0)),
        video_interval=float(service_intervals.get("video", 3.0)),
    )
