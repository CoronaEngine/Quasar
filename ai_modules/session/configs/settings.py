"""
会话管理配置
"""

from __future__ import annotations

from typing import Any, Dict

from ....ai_service.entrance import ai_entrance
# 网络请求配置
@ai_entrance.collector.register_setting("session")
def SESSION_SETTINGS() -> Dict[str, Any]:
    return {
        "ttl_seconds": 86400,  # 24 hours
        "max_sessions": 10000,
        "max_messages_per_session": 100,
        # 并发请求上限；<=0 表示不限制
        "max_concurrent_requests": 0,
        # 文件注册表线程池工作线程数；None 或 <=0 表示使用动态默认
        "file_registry_max_workers": None,
    }
