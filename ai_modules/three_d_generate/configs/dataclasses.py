from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=False)
class Rodin3DSettings:
    """
    Rodin 3D 生成配置
    """

    provider: str       # providers 里的 key
    base_url: str
    api_key: str

    # Rodin API endpoints（官方文档：/api/v2/rodin, /api/v2/status, /api/v2/download）
    generate_path: str 
    status_path: str 
    download_path: str 

    request_timeout: float = 300.0
    poll_interval: float = 2.0
    poll_timeout: float = 900.0  


@dataclass(frozen=False)
class Hunyuan3DSettings:
    """
    腾讯混元生3D 配置
    """

    provider: str = ""

    # API 密钥（Bearer Token）
    api_key: str = ""

    # 服务地域
    region: str = "ap-guangzhou"

    # API 域名
    endpoint: str = "tokenhub.tencentmaas.com"

    # 版本：pro（专业版） / rapid（极速版）
    version: str = "pro"

    # 默认结果格式：GLB, OBJ, STL, USDZ, FBX
    result_format: str = "GLB"

    # 是否开启 PBR 材质
    enable_pbr: bool = False

    # 模型版本（仅专业版）：3.0, 3.1
    model: str = "3.0"

    # 生成类型（仅专业版）：Normal, LowPoly, Geometry, Sketch
    generate_type: str = "Normal"

    # 面数（仅专业版），默认 500000
    face_count: int = 500000

    request_timeout: float = 300.0
    poll_interval: float = 3.0
    poll_timeout: float = 600.0
