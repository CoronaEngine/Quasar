"""
API 提供商配置 - 默认预设
"""

from __future__ import annotations

from typing import Any, Dict, List

# 默认提供商配置
from ....ai_service.entrance import ai_entrance
# 网络请求配置
@ai_entrance.collector.register_setting("providers")
def PROVIDERS() -> List[Dict[str, Any]]:
    return [
        {
            "name": "example",
            "type": "openai-compatible",
            "base_url": "https://api.example.com/v1",
            "api_key": "YOUR_API_KEY_HERE",
        },
    ]
