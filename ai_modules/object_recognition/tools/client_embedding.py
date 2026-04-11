"""
物体识别模块 —— 云端嵌入客户端（Dashscope SDK）

本模块通过 dashscope.MultiModalEmbedding SDK 调用通义嵌入服务，
不包含任何本地模型加载、GPU 推理或量化逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import logging
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..configs.dataclasses import RecognitionConfig

logger = logging.getLogger(__name__)

_CLIENT_INSTANCES: Dict[Tuple[str, str, Optional[int]], "DashscopeEmbeddingClient"] = {}
_CLIENT_LOCK = threading.Lock()

# Dashscope 支持的图片格式
_SUPPORTED_IMAGE_EXTS = frozenset(
    {"jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "tif"}
)


def _image_to_data_uri(image_path: str) -> str:
    """将本地图片文件转换为 base64 data URI。"""
    path = Path(image_path)
    ext = path.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def _build_input(
    image_paths: Optional[List[str]],
    text: Optional[str],
    instruction: Optional[str],
) -> List[dict]:
    """
    构造 Dashscope MultiModalEmbedding 的 input 列表。

    - instruction 非空时拼接到文本前缀（以空格分隔）
    - 单图用 "image" 字段，多图用 "multi_images" 字段
    - instruction 单独有效时直接作为文本传入
    """
    effective_text: str = ""
    if instruction and text:
        effective_text = f"{instruction} {text}"
    elif instruction:
        effective_text = instruction
    elif text:
        effective_text = text

    item: dict = {}
    if effective_text:
        item["text"] = effective_text

    if image_paths:
        data_uris = [_image_to_data_uri(p) for p in image_paths]
        if len(data_uris) == 1:
            item["image"] = data_uris[0]
        else:
            item["multi_images"] = data_uris

    return [item]


class DashscopeEmbeddingClient:
    """通过 Dashscope SDK 生成多模态嵌入向量。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        output_dim: Optional[int] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._output_dim = output_dim

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def embed_for_storage(
        self,
        image_paths: List[str],
        text: str = "",
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """生成存储侧（文档侧）嵌入向量。"""
        if not image_paths and not text:
            raise ValueError("存储嵌入至少需要提供图片或文字描述")
        input_data = _build_input(image_paths, text, instruction)
        return self._call(input_data, error_prefix="存储侧嵌入失败")

    def embed_for_query(
        self,
        image_paths: Optional[List[str]] = None,
        text: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """生成查询侧嵌入向量。"""
        if not image_paths and not text:
            raise ValueError("查询嵌入至少需要提供图片或文字描述")
        input_data = _build_input(image_paths, text, instruction)
        return self._call(input_data, error_prefix="查询侧嵌入失败")

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _call(self, input_data: list, error_prefix: str) -> np.ndarray:
        """调用 Dashscope MultiModalEmbedding 并解析返回向量。"""
        import dashscope

        kwargs: dict = {
            "api_key": self._api_key,
            "model": self._model,
            "input": input_data,
            "enable_fusion": True,
        }
        if self._output_dim is not None:
            kwargs["dimension"] = self._output_dim

        resp = dashscope.MultiModalEmbedding.call(**kwargs)

        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"{error_prefix}: status_code={resp.status_code}, "
                f"model={self._model}, message={getattr(resp, 'message', '')}"
            )

        return self._parse_response(resp, error_prefix)

    def _parse_response(self, resp, error_prefix: str) -> np.ndarray:
        """从 Dashscope 响应中提取并校验向量。"""
        try:
            raw_vec = resp.output["embeddings"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"{error_prefix}: 无法从响应中提取向量, output={resp.output}"
            ) from exc

        vec = np.array(raw_vec, dtype=np.float32)

        if vec.ndim != 1 or vec.shape[0] == 0:
            raise RuntimeError(
                f"{error_prefix}: 返回无效向量维度 shape={vec.shape}"
            )
        if self._output_dim is not None and vec.shape[0] != self._output_dim:
            raise RuntimeError(
                f"{error_prefix}: 返回维度 {vec.shape[0]} 与配置维度 {self._output_dim} 不一致"
            )
        return vec


class EmbeddingProvider(ABC):
    """嵌入向量提供者抽象接口。"""

    @abstractmethod
    def embed_for_storage(
        self,
        image_paths: List[str],
        text: str = "",
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def embed_for_query(
        self,
        image_paths: Optional[List[str]] = None,
        text: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        ...


class RemoteEmbeddingProvider(EmbeddingProvider):
    """基于 Dashscope SDK 的嵌入提供者。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        output_dim: Optional[int] = None,
    ) -> None:
        self._client = get_embedding_client(
            api_key=api_key,
            model=model,
            output_dim=output_dim,
        )

    def embed_for_storage(
        self,
        image_paths: List[str],
        text: str = "",
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        return self._client.embed_for_storage(
            image_paths=image_paths,
            text=text,
            instruction=instruction,
        )

    def embed_for_query(
        self,
        image_paths: Optional[List[str]] = None,
        text: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        return self._client.embed_for_query(
            image_paths=image_paths,
            text=text,
            instruction=instruction,
        )


def get_embedding_client(
    api_key: str,
    model: str,
    output_dim: Optional[int] = None,
) -> DashscopeEmbeddingClient:
    """获取 Dashscope 嵌入客户端单例（按 api_key/model/output_dim 缓存）。"""
    key = (api_key, model, output_dim)
    instance = _CLIENT_INSTANCES.get(key)
    if instance is not None:
        return instance

    with _CLIENT_LOCK:
        instance = _CLIENT_INSTANCES.get(key)
        if instance is None:
            instance = DashscopeEmbeddingClient(
                api_key=api_key,
                model=model,
                output_dim=output_dim,
            )
            _CLIENT_INSTANCES[key] = instance
            logger.info(
                "Dashscope 嵌入客户端已创建: model=%s, output_dim=%s",
                model,
                output_dim,
            )
    return instance


def build_provider(recognition_cfg: RecognitionConfig) -> EmbeddingProvider:
    """根据 RecognitionConfig 构建 Dashscope EmbeddingProvider 实例。"""
    logger.info(
        "使用 Dashscope 嵌入提供者: model=%s, dim=%d",
        recognition_cfg.dashscope_model,
        recognition_cfg.embedding.output_dim,
    )
    return RemoteEmbeddingProvider(
        api_key=recognition_cfg.dashscope_api_key,
        model=recognition_cfg.dashscope_model,
        output_dim=recognition_cfg.embedding.output_dim,
    )


__all__ = [
    "DashscopeEmbeddingClient",
    "EmbeddingProvider",
    "RemoteEmbeddingProvider",
    "get_embedding_client",
    "build_provider",
]
