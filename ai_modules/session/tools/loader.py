"""
会话配置加载器
"""

from typing import Any, Mapping

from ..configs.dataclasses import SessionConfig

from ....ai_service.entrance import ai_entrance

@ai_entrance.collector.register_loader('session')
def _load_session_config(raw: Mapping[str, Any] | None) -> SessionConfig:
    """加载会话配置"""
    if not isinstance(raw, Mapping):
        return SessionConfig()

    def _as_optional_int(v: Any) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return SessionConfig(
        ttl_seconds=int(raw.get("ttl_seconds", 86400)),
        max_sessions=int(raw.get("max_sessions", 10000)),
        max_messages_per_session=int(raw.get("max_messages_per_session", 100)),
        max_concurrent_requests=_as_optional_int(raw.get("max_concurrent_requests")),
        file_registry_max_workers=_as_optional_int(
            raw.get("file_registry_max_workers")
        ),
    )
