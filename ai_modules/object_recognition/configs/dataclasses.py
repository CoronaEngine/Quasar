"""
物体识别模块 —— 配置数据类

定义嵌入模型、向量数据库、识别参数等配置结构。
"""

from dataclasses import dataclass, field

from ai_config.paths_config import get_default_paths


def _default_assets_model_path() -> str:
    """从项目路径获取物体识别模型默认路径。"""
    return str(get_default_paths().assets_model_dir)


def _default_vector_db_path() -> str:
    """从项目路径获取物体识别数据库默认路径。"""
    return str(get_default_paths().object_recognition_db)


@dataclass(frozen=False)
class EmbeddingModelConfig:
    """云端 embedding 输出配置"""

    # 嵌入向量维度，须与远端服务保持一致
    output_dim: int = 1024


@dataclass(frozen=False)
class VectorDBConfig:
    """sqlite-vec 向量数据库配置"""

    # 数据库文件路径（单 .db 文件）
    db_path: str = field(default_factory=_default_vector_db_path)
    # 向量维度（须与 EmbeddingModelConfig.output_dim 一致）
    vector_dim: int = 1024
    # 搜索时默认返回的最大结果数
    default_top_k: int = 5


@dataclass(frozen=False)
class RecognitionConfig:
    """物体识别整体配置"""

    # 是否启用物体识别模块
    enable: bool = True
    # 嵌入模型配置
    embedding: EmbeddingModelConfig = \
        field(default_factory=EmbeddingModelConfig)
    # 向量数据库配置
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    # 每个物体的标准图片数量（六面图）
    standard_image_count: int = 6
    # 存储侧非对称指令
    storage_instruction: str = "Represent this document for retrieval:"
    # 查询侧非对称指令
    query_instruction: str = \
        "Represent the query for retrieving relevant documents:"

    # ── 云端嵌入服务配置（推荐通过 providers 统一管理）──
    # providers 中的 provider 名称
    provider: str = "dashscope"
    # Dashscope API Key（兼容字段；建议放到 providers 中）
    dashscope_api_key: str = ""
    # Dashscope 多模态嵌入模型名称
    dashscope_model: str = "tongyi-embedding-vision-plus-2026-03-06"

    # ── 目录自动扫描配置 ──
    # 扫描根目录路径（空字符串表示不扫描）
    auto_scan_dir: str = field(default_factory=_default_assets_model_path)
    # 是否自动嵌入并入库未登记的子文件夹（False 时仅输出警告）
    auto_scan_embed: bool = True
    # 每个子文件夹最多读取的图片数量
    auto_scan_max_images: int = 6


__all__ = [
    "EmbeddingModelConfig",
    "VectorDBConfig",
    "RecognitionConfig",
]
