"""
内存缓存实现（云端模式使用）

提供线程安全的内存缓存，支持：
- TTL 自动过期
- LRU 淘汰策略
- 定期清理
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import MEMORY_CACHE_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""

    data: str  # 存储的数据（base64 data URI 或 URL）
    resource_type: str  # 资源类型
    created_at: float  # 创建时间戳
    expire_at: float  # 过期时间戳
    session_id: str  # 关联的会话 ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """检查是否已过期"""
        return time.time() > self.expire_at

    @property
    def ttl_remaining(self) -> float:
        """剩余存活时间（秒）"""
        return max(0, self.expire_at - time.time())


class MemoryCache:
    """
    内存缓存（线程安全）

    用于云端模式暂存媒体数据，支持：
    - TTL 自动过期
    - LRU 淘汰策略
    - 定期清理
    """

    def __init__(
        self,
        default_ttl: int = MEMORY_CACHE_CONFIG["default_ttl_seconds"],
        max_items: int = MEMORY_CACHE_CONFIG["max_items"],
        cleanup_interval: int = MEMORY_CACHE_CONFIG["cleanup_interval"],
    ):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._max_items = max_items
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

        logger.info(f"MemoryCache 初始化: ttl={default_ttl}s, max_items={max_items}")

    def _generate_cache_id(self) -> str:
        """生成唯一的缓存 ID"""
        return f"cache_{uuid.uuid4().hex[:12]}"

    def _cleanup_expired(self) -> int:
        """清理过期条目（必须在锁内调用）"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return 0

        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]

        for key in expired_keys:
            del self._cache[key]

        self._last_cleanup = current_time

        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")

        return len(expired_keys)

    def _evict_if_needed(self) -> None:
        """如果超出容量限制，淘汰最早的条目（必须在锁内调用）"""
        while len(self._cache) >= self._max_items:
            # 找到最早创建的条目
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].created_at
            )
            del self._cache[oldest_key]
            logger.debug(f"淘汰缓存条目: {oldest_key}")

    def put(
        self,
        data: str,
        resource_type: str,
        session_id: str,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        存入缓存

        参数:
        - data: 要缓存的数据（base64 data URI 或 URL）
        - resource_type: 资源类型
        - session_id: 会话 ID
        - ttl: 存活时间（秒），None 使用默认值
        - metadata: 额外元数据

        返回:
        - cache_id: 缓存 ID
        """
        cache_id = self._generate_cache_id()
        current_time = time.time()
        effective_ttl = ttl if ttl is not None else self._default_ttl

        entry = CacheEntry(
            data=data,
            resource_type=resource_type,
            created_at=current_time,
            expire_at=current_time + effective_ttl,
            session_id=session_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._cleanup_expired()
            self._evict_if_needed()
            self._cache[cache_id] = entry

        logger.debug(
            f"缓存条目已创建: {cache_id}, type={resource_type}, "
            f"ttl={effective_ttl}s, size={len(data)} bytes"
        )

        return cache_id

    def get(self, cache_id: str) -> Optional[CacheEntry]:
        """
        获取缓存条目

        参数:
        - cache_id: 缓存 ID

        返回:
        - CacheEntry 或 None（如果不存在或已过期）
        """
        with self._lock:
            entry = self._cache.get(cache_id)
            if entry is None:
                return None
            if entry.is_expired:
                del self._cache[cache_id]
                return None
            return entry

    def delete(self, cache_id: str) -> bool:
        """删除缓存条目"""
        with self._lock:
            if cache_id in self._cache:
                del self._cache[cache_id]
                return True
            return False

    def clear_session(self, session_id: str) -> int:
        """清理指定会话的所有缓存"""
        with self._lock:
            keys_to_delete = [
                key
                for key, entry in self._cache.items()
                if entry.session_id == session_id
            ]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_size = sum(len(e.data) for e in self._cache.values())
            expired_count = sum(1 for e in self._cache.values() if e.is_expired)
            return {
                "total_items": len(self._cache),
                "expired_items": expired_count,
                "total_size_bytes": total_size,
                "max_items": self._max_items,
                "default_ttl": self._default_ttl,
            }


# ==============================================================================
# 全局单例
# ==============================================================================

_memory_cache: Optional[MemoryCache] = None
_memory_cache_lock = threading.Lock()


def get_memory_cache() -> MemoryCache:
    """获取内存缓存单例"""
    global _memory_cache
    if _memory_cache is None:
        with _memory_cache_lock:
            if _memory_cache is None:
                _memory_cache = MemoryCache()
    return _memory_cache


def reset_memory_cache() -> None:
    """重置内存缓存（用于测试）"""
    global _memory_cache
    with _memory_cache_lock:
        _memory_cache = None


__all__ = [
    "CacheEntry",
    "MemoryCache",
    "get_memory_cache",
    "reset_memory_cache",
]
