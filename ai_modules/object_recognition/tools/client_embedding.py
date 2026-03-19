"""
物体识别模块 —— Qwen3-VL-Embedding 嵌入客户端

封装 Qwen3-VL-Embedding 模型的加载、推理、向量融合逻辑。
支持:
- 8B / 2B 两种模型尺寸
- 4-bit nf4 量化 + Flash Attention 2
- Matryoshka 维度切换 (1024 / 512 / 256)
- 多图 + 文本的混合输入融合为单一嵌入向量
- 非对称指令（存储侧 vs 查询侧）

依赖:
    pip install torch transformers bitsandbytes accelerate
"""

from __future__ import annotations

import os
import logging
import threading
from typing import List, Optional

import numpy as np

from ai_modules.object_recognition.configs.dataclasses import EmbeddingModelConfig

logger = logging.getLogger(__name__)

# 禁用 Hugging Face Hub 联网请求，使用本地缓存的模型文件
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 全局模型单例及其锁
_EMBEDDER_INSTANCE: Optional["Qwen3VLEmbeddingClient"] = None
_EMBEDDER_LOCK = threading.Lock()


class Qwen3VLEmbeddingClient:
    """
    Qwen3-VL-Embedding 嵌入模型客户端。

    全局单例，首次调用时加载模型，后续复用。
    支持将多张图片 + 可选文本融合为单一嵌入向量。
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        """
        初始化嵌入模型客户端。

        参数:
            config: 嵌入模型配置
        """
        self.config = config
        self._model = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load_model(self) -> None:
        """懒加载模型（首次调用时触发）"""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            logger.info(
                f"正在加载 Qwen3-VL-Embedding 模型: "
                f"size={self.config.model_size}, "
                f"path={self.config.model_path}, "
                f"4bit={self.config.use_4bit}, "
                f"flash_attn={self.config.use_flash_attention}"
            )

            try:
                import torch
                from ai_tools.qwen3_vl_embedding import Qwen3VLEmbedder

                # 构建量化配置
                model_kwargs = {
                    "device_map": self.config.device_map,
                    "torch_dtype": torch.bfloat16,
                }

                # 4-bit 量化配置
                if self.config.use_4bit:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("已启用 4-bit nf4 量化 (double_quant + bfloat16)")

                # Flash Attention 2
                if self.config.use_flash_attention:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("已启用 Flash Attention 2")

                # 加载模型
                self._model = Qwen3VLEmbedder(
                    model_name_or_path=self.config.model_path,
                    **model_kwargs,
                )

                logger.info(
                    f"Qwen3-VL-Embedding 模型加载完成 "
                    f"(output_dim={self.config.output_dim})"
                )

            except ImportError as e:
                raise RuntimeError(
                    f"无法导入 Qwen3-VL-Embedding: {e}\n"
                    f"请确认已安装所需依赖: pip install torch transformers bitsandbytes accelerate"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"加载 Qwen3-VL-Embedding 模型失败: {e}"
                ) from e

    def embed_for_storage(
        self,
        image_paths: List[str],
        text: str = "",
    ) -> np.ndarray:
        """
        存储侧嵌入：将多张图片（+ 可选文本描述）融合为单一向量。

        使用非对称指令 "Represent this document for retrieval:"

        参数:
            image_paths: 图片路径列表（最多 6 张六面图）
            text:        可选的文字描述

        返回:
            归一化后的嵌入向量 (numpy array, shape=[output_dim])
        """
        self._load_model()

        if not image_paths and not text:
            raise ValueError("存储嵌入至少需要提供图片或文字描述")

        # 构建输入 —— 所有图片 + 文本放入同一个 dict 的 image/text 字段
        input_dict = {
            "instruction": "Represent this document for retrieval:",
        }

        # 图片列表（支持 0~6 张）
        if image_paths:
            input_dict["image"] = self._resolve_image_paths(image_paths)

        # 文本描述
        if text:
            input_dict["text"] = text

        # 调用模型推理（返回归一化后的 torch.Tensor）
        with self._inference_lock:
            embeddings = self._model.process([input_dict], normalize=True)

        # 取第一个结果，转为 numpy
        vec = embeddings[0].cpu().float().numpy()

        # Matryoshka 维度截断（取前 output_dim 维）
        if self.config.output_dim < vec.shape[0]:
            vec = vec[: self.config.output_dim]

        # 截断后重新归一化
        vec = self._normalize(vec)

        logger.debug(
            f"存储侧嵌入完成: images={len(image_paths)}, "
            f"text_len={len(text)}, dim={vec.shape}"
        )
        return vec

    def embed_for_query(
        self,
        image_paths: Optional[List[str]] = None,
        text: Optional[str] = None,
    ) -> np.ndarray:
        """
        查询侧嵌入：将查询图片（+ 可选文本）融合为查询向量。

        使用非对称指令 "Represent the query for retrieving relevant documents:"

        参数:
            image_paths: 查询图片路径列表（0~6 张）
            text:        查询文字描述

        返回:
            归一化后的查询向量 (numpy array, shape=[output_dim])
        """
        self._load_model()

        if not image_paths and not text:
            raise ValueError("查询嵌入至少需要提供图片或文字描述")

        input_dict = {
            "instruction": "Represent the query for retrieving relevant documents:",
        }

        if image_paths:
            input_dict["image"] = self._resolve_image_paths(image_paths)

        if text:
            input_dict["text"] = text

        with self._inference_lock:
            embeddings = self._model.process([input_dict], normalize=True)

        vec = embeddings[0].cpu().float().numpy()

        # Matryoshka 维度截断
        if self.config.output_dim < vec.shape[0]:
            vec = vec[: self.config.output_dim]

        vec = self._normalize(vec)

        logger.debug(
            f"查询侧嵌入完成: images={len(image_paths or [])}, "
            f"text_len={len(text or '')}, dim={vec.shape}"
        )
        return vec

    # ------------------------------------------------------------------ #
    #  内部辅助方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_image_paths(image_paths: List[str]) -> List[str]:
        """
        解析图片路径，确保格式正确。

        支持:
        - 本地文件路径（自动转为绝对路径并验证存在性）
        - HTTP(S) URL（直接透传）
        - data: URI（直接透传）
        """
        resolved = []
        for path in image_paths:
            if not path:
                continue
            # HTTP(S) URL 或 data: URI 直接使用
            if path.startswith(("http://", "https://", "data:")):
                resolved.append(path)
            else:
                # 本地文件路径
                abs_path = os.path.abspath(path)
                if not os.path.isfile(abs_path):
                    logger.warning(f"图片文件不存在，已跳过: {abs_path}")
                    continue
                resolved.append(abs_path)

        if not resolved:
            logger.warning("解析后无有效图片路径")

        return resolved

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2 归一化向量"""
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            logger.warning("嵌入向量范数接近零，跳过归一化")
            return vec
        return vec / norm


def get_embedding_client(config: EmbeddingModelConfig) -> Qwen3VLEmbeddingClient:
    """
    获取全局嵌入模型客户端单例（线程安全）。

    参数:
        config: 嵌入模型配置

    返回:
        Qwen3VLEmbeddingClient 单例实例
    """
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER_INSTANCE is None:
                _EMBEDDER_INSTANCE = Qwen3VLEmbeddingClient(config)
                logger.info("嵌入模型客户端单例已创建")
    return _EMBEDDER_INSTANCE


__all__ = [
    "Qwen3VLEmbeddingClient",
    "get_embedding_client",
]
