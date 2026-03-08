"""
物体识别模块 —— 配置数据类

定义嵌入模型、向量数据库、识别参数等配置结构。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=False)
class EmbeddingModelConfig:
    """Qwen3-VL-Embedding 嵌入模型配置"""

    # 模型尺寸：8B 或 2B
    model_size: str = "2B"
    # 嵌入向量维度（Matryoshka 支持 1024/512/256）
    output_dim: int = 1024
    # 模型权重路径（本地路径或 HuggingFace 仓库 ID）
    model_path: str = "Qwen/Qwen3-VL-Embedding-2B"
    # 是否启用 4-bit 量化（nf4 + double_quant + bfloat16）
    use_4bit: bool = True
    # 是否启用 Flash Attention 2
    use_flash_attention: bool = False
    # 设备映射策略
    device_map: str = "auto"


@dataclass(frozen=False)
class VectorDBConfig:
    """sqlite-vec 向量数据库配置"""

    # 数据库文件路径（单 .db 文件）
    db_path: str = "./object_recognition.db"
    # 向量维度（须与 EmbeddingModelConfig.output_dim 一致）
    vector_dim: int = 1024
    # 搜索时默认返回的最大结果数
    default_top_k: int = 5


@dataclass(frozen=False)
class RecognitionConfig:
    """物体识别整体配置"""

    # 是否启用物体识别模块
    enable: bool = False
    # 嵌入模型配置
    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    # 向量数据库配置
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    # 每个物体的标准图片数量（六面图）
    standard_image_count: int = 6
    # 存储侧非对称指令
    storage_instruction: str = "Represent this document for retrieval:"
    # 查询侧非对称指令
    query_instruction: str = "Represent the query for retrieving relevant documents:"

    # ── 目录自动扫描配置 ──
    # 扫描根目录路径（空字符串表示不扫描）
    auto_scan_dir: str = ""
    # 是否自动嵌入并入库未登记的子文件夹（False 时仅输出警告）
    auto_scan_embed: bool = False
    # 每个子文件夹最多读取的图片数量
    auto_scan_max_images: int = 6


__all__ = [
    "EmbeddingModelConfig",
    "VectorDBConfig",
    "RecognitionConfig",
]
