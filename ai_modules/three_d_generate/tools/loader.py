"""
多模态理解配置加载器
"""

import logging
from typing import Any, Mapping

from ..configs.dataclasses import Rodin3DSettings, Hunyuan3DSettings
from ....ai_service.entrance import ai_entrance
from ....ai_tools.helpers import _as_bool,_as_float

logger = logging.getLogger(__name__)

logger.debug("加载 Rodin 3D 配置加载器")
@ai_entrance.collector.register_loader("rodin3d")
def _load_rodin_3d_config(raw: Mapping[str, Any] | None) -> Rodin3DSettings:
    """加载多模态理解配置"""

    logger.debug("加载 Rodin 3D 配置加载器")
    if not isinstance(raw, Mapping):
        return Rodin3DSettings()

    return Rodin3DSettings(
        provider=raw.get("provider",''),       # providers 里的 key
        base_url=raw.get("base_url"),
        api_key=raw.get("api_key"),

        # Rodin API endpoints（官方文档：/api/v2/rodin, /api/v2/status, /api/v2/download）
        generate_path=raw.get("generate_path"), 
        status_path=raw.get("status_path"), 
        download_path=raw.get("download_path"), 

        request_timeout=_as_float(raw.get("request_timeout"), 300.0),
        poll_interval=_as_float(raw.get("poll_interval"), 2.0),
        poll_timeout=_as_float(raw.get("poll_timeout"), 900.0)
    )


logger.debug("加载混元 3D 配置加载器")
@ai_entrance.collector.register_loader("hunyuan3d")
def _load_hunyuan_3d_config(raw: Mapping[str, Any] | None) -> Hunyuan3DSettings:
    """加载混元3D配置"""

    logger.debug("加载混元 3D 配置加载器")
    if not isinstance(raw, Mapping):
        return Hunyuan3DSettings()

    # 支持多 AK: api_keys 列表优先, 单 api_key 兜底
    api_keys_raw = raw.get("api_keys", [])
    if isinstance(api_keys_raw, list) and api_keys_raw:
        api_keys = [str(k).strip() for k in api_keys_raw if str(k).strip()]
    else:
        api_keys = []
    single_key = raw.get("api_key", "") or ""

    return Hunyuan3DSettings(
        enable=_as_bool(raw.get("enable"), False),
        provider=raw.get("provider", ""),
        api_key=single_key,
        api_keys=api_keys,
        max_concurrent_generations=int(raw.get("max_concurrent_generations", 3) or 3),
        region=raw.get("region", "ap-guangzhou"),
        endpoint=raw.get("endpoint", "tokenhub.tencentmaas.com"),
        version=raw.get("version", "pro"),
        result_format=raw.get("result_format", "GLB"),
        enable_pbr=_as_bool(raw.get("enable_pbr"), False),
        model=raw.get("model", "3.0"),
        generate_type=raw.get("generate_type", "Normal"),
        face_count=int(raw.get("face_count", 500000)),
        request_timeout=_as_float(raw.get("request_timeout"), 300.0),
        poll_interval=_as_float(raw.get("poll_interval"), 3.0),
        poll_timeout=_as_float(raw.get("poll_timeout"), 600.0),
    )
