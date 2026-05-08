"""
网络和轮询配置
"""

from __future__ import annotations

from typing import Any, Dict
from ....ai_service.entrance import ai_entrance

# 异步任务轮询配置
@ai_entrance.collector.register_setting("polling")
def POLLING_SETTINGS() -> Dict[str, Any]:
    return {
        "max_wait_seconds": 150,
        "default_interval": 3.0,
        "service_intervals": {
            "speech": 2.0,
            "music": 5.0,
            "video": 3.0,
        },
    }
