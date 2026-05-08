"""
Provider 配置加载器
"""

import os
from typing import Any, Dict, Mapping

from ..configs.dataclasses import ProviderConfig
from ....ai_service.entrance import ai_entrance


@ai_entrance.collector.register_loader('providers')
def _load_providers(raw: Any) -> Dict[str, ProviderConfig]:
    """加载 Provider 配置"""
    providers: Dict[str, ProviderConfig] = {}
    entries = raw if isinstance(raw, list) else []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not name:
            continue
        api_key = entry.get("api_key")
        api_key_env = entry.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(str(api_key_env), api_key)
        headers = (
            entry.get("headers") if isinstance(entry.get("headers"), Mapping) else {}
        )
        providers[name] = ProviderConfig(
            name=name,
            type=str(entry.get("type", "openai")),
            base_url=entry.get("base_url"),
            api_key=api_key,
            headers={str(k): str(v) for k, v in headers.items()},
        )
    return providers
