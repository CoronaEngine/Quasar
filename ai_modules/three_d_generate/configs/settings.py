"""3D生成配置和提示词"""
from __future__ import annotations

from typing import Any, Dict

from ....ai_service.entrance import ai_entrance


@ai_entrance.collector.register_setting("rodin3d")
def RODIN_3D_SETTINGS() -> Dict[str, Any]:
    # 模型文件本地保存策略由 three_d_generate.tools.model_tools 统一管理：
    # 固定写入 <repo_root>/assets/model/<object_id>/，不依赖 download_dir 或环境变量。
    return {
        "provider": "hyper3d_rodin",
        "base_url": "https://api.hyper3d.com",
        "api_key": "",

        "generate_path": "/api/v2/rodin",
        "download_path": "/api/v2/download",
        "status_path": "/api/v2/status",
        "request_timeout": 300.0,
        "poll_interval": 2.0,
        "poll_timeout": 900.0,
    }


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
