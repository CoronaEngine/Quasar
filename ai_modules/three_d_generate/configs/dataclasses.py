from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=False)
class Hunyuan3DSettings:
    """
    腾讯混元生3D 配置
    """

    enable: bool = False
    provider: str = ""

    # API 密钥（Bearer Token）
    api_key: str = ""

    # 多 AK 并行: 每个 key 独立建 client, 提高并发上限
    api_keys: List[str] = field(default_factory=list)

    # 单 AK 最大并发任务数 (混元限制 3)
    max_concurrent_generations: int = 3

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
