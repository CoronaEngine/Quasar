"""3D生成配置和提示词"""
from __future__ import annotations

from typing import Any, Dict

from ....ai_service.entrance import ai_entrance

@ai_entrance.collector.register_setting("hunyuan3d")
def HUNYUAN_3D_SETTINGS() -> Dict[str, Any]:
    return {
        "enable": False,
        "api_key": "your_api_key",
        "region": "ap-guangzhou",
        "endpoint": "tokenhub.tencentmaas.com",
        "version": "pro",           # pro（专业版） / rapid（极速版）
        "result_format": "GLB",      # GLB, OBJ, STL, USDZ, FBX
        "enable_pbr": False,
        "model": "3.0",             # 专业版模型版本：3.0, 3.1
        "generate_type": "Normal",   # Normal, LowPoly, Geometry, Sketch
        "face_count": 500000,
        "request_timeout": 300.0,
        "poll_interval": 3.0,
        "poll_timeout": 600.0,
    }
