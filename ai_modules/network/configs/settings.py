"""
网络和轮询配置
"""

from __future__ import annotations

from typing import Any, Dict
from ....ai_service.entrance import ai_entrance
# 网络请求配置
@ai_entrance.collector.register_setting("network")
def NETWORK_SETTINGS() -> Dict[str, Any]:
    return  {
        "request_timeout": 60,
        "download_timeout": 300,
        "download_chunk_size": 8192,
        "download_retries": 2,
        "download_backoff_factor": 0.5,
    }

