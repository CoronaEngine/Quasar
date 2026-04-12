"""
物体识别模块 —— 云端嵌入客户端（Dashscope SDK）

本模块通过 dashscope.MultiModalEmbedding SDK 调用通义嵌入服务，
不包含任何本地模型加载、GPU 推理或量化逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from io import BytesIO
import logging
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from ..configs.dataclasses import RecognitionConfig

logger = logging.getLogger(__name__)

_CLIENT_INSTANCES: Dict[Tuple[str, str, Optional[int]], "DashscopeEmbeddingClient"] = {}
_CLIENT_LOCK = threading.Lock()

# Dashscope 支持的图片格式
_SUPPORTED_IMAGE_EXTS = frozenset(
    {"jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "tif"}
)
_MAX_IMAGE_BYTES = 1024 * 1024
_MIN_IMAGE_SIDE = 256
_JPEG_QUALITY_STEPS = (90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30)
_RESIZE_SCALE_STEPS = (1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2)


def _normalize_image_ext(ext: str) -> str:
    """规范化图片扩展名。"""
    normalized = ext.lower().lstrip(".")
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in _SUPPORTED_IMAGE_EXTS:
        raise ValueError(f"不支持的图片格式: {ext}")
    return normalized


def _ensure_rgb_image(image: Image.Image) -> Image.Image:
    """将图片转换为适合 JPEG 压缩的 RGB。"""
    if image.mode == "RGB":
        return image

    has_alpha = image.mode in {"RGBA", "LA"}
    has_palette_alpha = image.mode == "P" and "transparency" in image.info
    if has_alpha or has_palette_alpha:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, image.convert("RGBA"))
        return composited.convert("RGB")

    return image.convert("RGB")


def _encode_jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    """将图片编码为 JPEG 二进制。"""
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
    )
    return buffer.getvalue()


def _compress_image_bytes(path: Path, original_size: int) -> Tuple[bytes, str]:
    """将图片压缩到 1MB 以内，返回压缩后的二进制和扩展名。"""
    with Image.open(path) as source:
        source.load()
        working = ImageOps.exif_transpose(source)

    working = _ensure_rgb_image(working)
    base_width, base_height = working.size
    best_bytes = b""

    for scale in _RESIZE_SCALE_STEPS:
        resized_width = max(_MIN_IMAGE_SIDE, int(base_width * scale))
        resized_height = max(_MIN_IMAGE_SIDE, int(base_height * scale))
        resized_size = (resized_width, resized_height)

        if resized_size == working.size:
            candidate_image = working
        else:
            candidate_image = working.resize(
                resized_size,
                Image.Resampling.LANCZOS,
            )

        for quality in _JPEG_QUALITY_STEPS:
            candidate_bytes = _encode_jpeg_bytes(candidate_image, quality)
            if not best_bytes or len(candidate_bytes) < len(best_bytes):
                best_bytes = candidate_bytes
            if len(candidate_bytes) <= _MAX_IMAGE_BYTES:
                logger.info(
                    "嵌入图片已压缩: path=%s, original_bytes=%s, compressed_bytes=%s, scale=%.2f, quality=%s",
                    path,
                    original_size,
                    len(candidate_bytes),
                    scale,
                    quality,
                )
                return candidate_bytes, "jpeg"

    compressed_size = len(best_bytes)
    raise RuntimeError(
        f"图片压缩失败: path={path}, original_bytes={original_size}, compressed_bytes={compressed_size}"
    )


def _read_image_payload(path: Path) -> Tuple[bytes, str]:
    """读取图片二进制；超出 1MB 时自动压缩。"""
    ext = _normalize_image_ext(path.suffix)
    raw_bytes = path.read_bytes()
    if len(raw_bytes) <= _MAX_IMAGE_BYTES:
        return raw_bytes, ext
    return _compress_image_bytes(path, len(raw_bytes))


def _image_to_data_uri(image_path: str) -> str:
    """将本地图片文件转换为 base64 data URI。"""
    path = Path(image_path)
    payload, ext = _read_image_payload(path)
    b64 = base64.b64encode(payload).decode("utf-8")
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
