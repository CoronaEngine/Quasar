"""
云端存储适配器

使用内存缓存存储媒体数据，返回 cache:// 协议的 URL。
支持 TTL 自动过期和 LRU 淘汰策略。

安全性：
- 云端 URL 会被下载并转换为 base64 存储，避免泄露上游 API URL
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx

from .adapter_base import (
    StorageAdapter,
    normalize_to_data_uri,
)
from .cache import get_memory_cache
from .config import MEMORY_CACHE_CONFIG
from .result import StorageResult

logger = logging.getLogger(__name__)


class CloudStorageAdapter(StorageAdapter):
    """
    云端存储适配器

    使用内存缓存存储媒体数据，返回 cache:// 协议的 URL。

    存储流程：
    1. 接收 base64 数据或云端 URL
    2. 存入内存缓存，生成 cache_id
    3. 返回 cache://{cache_id} 格式的 URL

    解析流程：
    1. 接收 cache://{cache_id} URL
    2. 从内存缓存获取数据
    3. 返回 base64 data URI（安全，不泄露上游 URL）
    """

    def __init__(self, ttl_seconds: int = MEMORY_CACHE_CONFIG["default_ttl_seconds"]):
        """
        初始化云端存储适配器

        参数:
        - ttl_seconds: 缓存过期时间（秒），默认 1 小时
        """
        self._cache = get_memory_cache()
        self._default_ttl = ttl_seconds
        logger.info(f"CloudStorageAdapter 初始化，默认 TTL: {ttl_seconds}s")

    def _download_to_base64(
        self,
        url: str,
        resource_type: str,
        timeout: int = 150,
    ) -> str:
        """
        下载 URL 内容并转换为 base64 data URI

        参数:
        - url: 要下载的 URL
        - resource_type: 资源类型（image/video/audio）
        - timeout: 下载超时时间（秒）

        返回:
        - base64 data URI (data:mime/type;base64,...)
        """
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

            # 从 Content-Type 获取 MIME 类型
            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if not content_type:
                # 根据资源类型推断默认 MIME 类型
                default_mime_types = {
                    "image": "image/png",
                    "video": "video/mp4",
                    "audio": "audio/mpeg",
                }
                content_type = default_mime_types.get(resource_type, "application/octet-stream")

            # 转换为 base64
            b64_data = base64.b64encode(response.content).decode("utf-8")
            return f"data:{content_type};base64,{b64_data}"

    def save_from_url(
        self,
        cloud_url: str,
        session_id: str,
        resource_type: str,
        original_name: Optional[str] = None,
        url_expire_time: Optional[int] = None,
    ) -> StorageResult:
        """
        云端模式：下载云端 URL 内容并转为 base64 存入缓存

        安全性：下载内容转为 base64 存储，避免泄露上游 API URL
        """
        ttl = self._calculate_ttl(url_expire_time)

        try:
            # 下载并转换为 base64 data URI
            data_uri = self._download_to_base64(cloud_url, resource_type)

            cache_id = self._cache.put(
                data=data_uri,
                resource_type=resource_type,
                session_id=session_id,
                ttl=ttl,
                metadata={
                    "source": "url_to_base64",
                    "original_name": original_name,
                    "original_url": cloud_url,  # 保留原始 URL，供需要时使用
                },
            )

            expire_dt = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            cache_url = f"cache://{cache_id}"
            logger.debug(
                f"云端 URL 已下载并缓存为 base64: {cloud_url[:50]}... -> {cache_url}"
            )

            return StorageResult(url=cache_url, url_expire_time=int(expire_dt.timestamp()))

        except Exception as e:
            logger.error(f"下载云端 URL 失败: {cloud_url[:50]}... - {e}")
            raise

    def save_from_base64(
        self,
        data_uri: str,
        session_id: str,
        resource_type: str,
        filename_prefix: str = "resource",
        url_expire_time: Optional[int] = None,
    ) -> StorageResult:
        """云端模式：将 base64 数据存入缓存"""
        normalized_data = normalize_to_data_uri(data_uri, resource_type)

        cache_id = self._cache.put(
            data=normalized_data,
            resource_type=resource_type,
            session_id=session_id,
            ttl=self._default_ttl,
            metadata={
                "source": "base64",
                "filename_prefix": filename_prefix,
            },
        )

        expire_dt = datetime.now(timezone.utc) + timedelta(seconds=self._default_ttl)
        cache_url = f"cache://{cache_id}"
        logger.debug(
            f"Base64 数据已缓存: {len(data_uri)} bytes -> {cache_url}, "
            f"TTL={self._default_ttl}s"
        )

        return StorageResult(url=cache_url, url_expire_time=int(expire_dt.timestamp()))

    def resolve_cache_url(
        self, cache_url: str, return_original_url: bool = False
    ) -> Optional[str]:
        """
        解析 cache:// URL，返回缓存的数据

        参数:
        - cache_url: cache://{cache_id} 格式的 URL
        - return_original_url: 是否返回原始云端 URL
          - False (默认): 返回 base64 data URI（安全，不泄露上游 URL）
          - True: 返回原始云端 URL（用于需要 HTTP URL 的上游 API）

        返回:
        - base64 data URI 或原始 URL，如果已过期返回 None
        """
        if not cache_url.startswith("cache://"):
            return cache_url

        cache_id = cache_url[8:]
        entry = self._cache.get(cache_id)

        if entry is None:
            logger.warning(f"缓存条目不存在或已过期: {cache_id}")
            return None

        # 如果需要原始 URL 且存在，返回原始 URL
        if return_original_url:
            original_url = entry.metadata.get("original_url")
            if original_url:
                return original_url
            # 如果没有原始 URL（如从 base64 直接存入），返回 data URI
            logger.debug(f"缓存条目没有原始 URL，返回 data URI: {cache_id}")

        return entry.data

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self._cache.get_stats()

    def clear_session_cache(self, session_id: str) -> int:
        """清理指定会话的缓存"""
        return self._cache.clear_session(session_id)

    def _calculate_ttl(self, url_expire_time: Optional[int]) -> int:
        """计算实际 TTL（取默认值和 URL 过期时间的较小值）"""
        ttl = self._default_ttl
        if url_expire_time:
            try:
                # url_expire_time 是秒级时间戳
                current_timestamp = int(datetime.now(timezone.utc).timestamp())
                url_ttl = url_expire_time - current_timestamp
                if url_ttl > 0:
                    ttl = min(ttl, int(url_ttl))
            except (ValueError, TypeError):
                pass
        return ttl


__all__ = ["CloudStorageAdapter"]
