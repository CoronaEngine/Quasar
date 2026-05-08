"""
存储适配器基类与工具函数
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from .result import StorageResult

logger = logging.getLogger(__name__)


# ==============================================================================
# 工具函数
# ==============================================================================


def normalize_to_data_uri(data: str, resource_type: str) -> str:
    """
    将 base64 数据标准化为 data URI 格式

    参数:
    - data: 可能是纯 base64 或完整 data URI
    - resource_type: 资源类型 (image/video/audio)

    返回:
    - 完整的 data URI (data:mime/type;base64,...)

    注意:
    - 如果传入的是 HTTP/HTTPS URL，会直接返回（不做转换）
    - 如果传入的是 file:// URL，会直接返回（不做转换）
    """
    # 如果已经是 data URI，直接返回
    if data.startswith("data:"):
        return data

    # 如果是 URL（http/https/file），直接返回，不要错误地转成 base64
    if data.startswith(("http://", "https://", "file://")):
        return data

    # 根据资源类型推断默认 MIME 类型
    default_mime_types = {
        "image": "image/png",
        "video": "video/mp4",
        "audio": "audio/mpeg",
    }
    mime_type = default_mime_types.get(resource_type, "application/octet-stream")

    # 尝试从 base64 数据推断图片格式
    if resource_type == "image" and len(data) > 10:
        if data.startswith("iVBORw0KGgo"):
            mime_type = "image/png"
        elif data.startswith("/9j/"):
            mime_type = "image/jpeg"
        elif data.startswith("R0lGOD"):
            mime_type = "image/gif"
        elif data.startswith("UklGR"):
            mime_type = "image/webp"

    return f"data:{mime_type};base64,{data}"


# ==============================================================================
# 存储适配器抽象基类
# ==============================================================================


class StorageAdapter(ABC):
    """存储适配器抽象基类"""

    @abstractmethod
    def save_from_url(
        self,
        cloud_url: str,
        session_id: str,
        resource_type: str,
        original_name: Optional[str] = None,
        url_expire_time: Optional[int] = None,
    ) -> StorageResult:
        """
        保存资源并返回可访问的 URL

        参数:
        - cloud_url: 云端资源 URL
        - session_id: 会话 ID
        - resource_type: 资源类型（image/video/audio）
        - original_name: 原始文件名（可选）
        - url_expire_time: URL 过期时间（秒级时间戳，可选）

        返回:
        - StorageResult: 包含 url 和 url_expire_time 的结果对象
        """
        pass

    @abstractmethod
    def save_from_base64(
        self,
        data_uri: str,
        session_id: str,
        resource_type: str,
        filename_prefix: str = "resource",
        url_expire_time: Optional[int] = None,
    ) -> StorageResult:
        """
        保存 base64 数据并返回可访问的 URL

        参数:
        - data_uri: base64 数据 URI (data:image/xxx;base64,...)
        - session_id: 会话 ID
        - resource_type: 资源类型（image/video/audio）
        - filename_prefix: 文件名前缀
        - url_expire_time: URL 过期时间（秒级时间戳，可选）

        返回:
        - StorageResult: 包含 url 和 url_expire_time 的结果对象
        """
        pass


__all__ = [
    "StorageAdapter",
    "normalize_to_data_uri",
]
