"""
网络配置加载器
"""

from typing import Any, Mapping
from ..configs.dataclasses import NetworkConfig
from ....ai_service.entrance import ai_entrance

@ai_entrance.collector.register_loader('network')
def _load_network_config(raw: Mapping[str, Any] | None) -> NetworkConfig:
    """加载网络配置"""
    if not isinstance(raw, Mapping):
        return NetworkConfig()
    return NetworkConfig(
        request_timeout=int(raw.get("request_timeout", 60)),
        download_timeout=int(raw.get("download_timeout", 300)),
        download_chunk_size=int(raw.get("download_chunk_size", 8192)),
        download_retries=int(raw.get("download_retries", 2)),
        download_backoff_factor=float(raw.get("download_backoff_factor", 0.5)),
    )
