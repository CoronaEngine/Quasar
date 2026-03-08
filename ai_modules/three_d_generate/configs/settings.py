"""3D生成配置和提示词"""
from __future__ import annotations

from typing import Any, Dict

from ai_service.entrance import ai_entrance
@ai_entrance.collector.register_setting("rodin3d")
def RODIN_3D_SETTINGS() -> Dict[str, Any]:
    # 模型文件本地保存策略由 three_d_generate.tools.model_tools 统一管理：
    # 固定写入 <repo_root>/assets/model/<object_id>/，不依赖 download_dir 或环境变量。
    return {
        "base_url": "https://api.hyper3d.com",
        "api_key": "your key",
       
        "generate_path": "/api/v2/rodin",
        "download_path": "/api/v2/download",
        "status_path": "/api/v2/status",
        "request_timeout": 300.0,
        "poll_interval": 2.0,
        "poll_timeout": 900.0,
    }


