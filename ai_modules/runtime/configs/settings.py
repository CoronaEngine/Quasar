"""
运行时基础配置
"""

from __future__ import annotations

from typing import Any, Dict

from ....ai_service.entrance import ai_entrance

@ai_entrance.collector.register_setting("runtime")
def RUNTIME_SETTINGS() -> Dict[str, Any]:
    return {
        "enable_gpu": False,
        "log_level": "INFO",
    }
