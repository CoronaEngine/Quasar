"""
存储适配器工厂

根据运行模式自动选择本地、云端或 OSS 存储适配器。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from .adapter_base import StorageAdapter
from .adapter_cloud import (
    CloudStorageAdapter,
)
from .adapter_local import (
    LocalStorageAdapter,
)
from .config import MEMORY_CACHE_CONFIG

logger = logging.getLogger(__name__)


def _try_import_oss_adapter():
    """
    尝试导入 OSS 存储适配器

    OSS 适配器位于私密仓库 oss_storage 中，
    如果导入失败则返回 None。

    检测顺序：
    1. 检查 alibabacloud_oss_v2 SDK 是否安装
    2. 检查 oss_storage 模块是否存在
    """
    # Step 1: 检查 OSS SDK
    try:
        import alibabacloud_oss_v2

        logger.info(
            f"✓ OSS SDK 已安装 (alibabacloud_oss_v2 v{getattr(alibabacloud_oss_v2, '__version__', 'unknown')})"
        )
    except ImportError as e:
        logger.warning(
            f"✗ OSS SDK 未安装，将使用 CloudStorageAdapter（返回 base64）\n"
            f"  导入错误: {e}\n"
            f"  提示: 请运行 pip install alibabacloud-oss-v2"
        )
        return None

    # Step 2: 检查 OSS 存储组件
    try:
        from file_storage.oss_storage.adapter import OSSStorageAdapter

        logger.info("✓ OSS 存储组件已安装 (oss_storage)")
        return OSSStorageAdapter
    except ImportError as e:
        logger.warning(
            f"✗ OSS 存储组件未安装，将使用 CloudStorageAdapter（返回 base64）\n"
            f"  导入错误: {e}\n"
            f"  提示: 请确保 InnerAgentWorkflow 子模块已正确克隆到 Docker 镜像中"
        )
        return None


_adapter_instance: Optional[StorageAdapter] = None
_adapter_lock = threading.Lock()


def get_storage_adapter() -> StorageAdapter:
    """
    根据运行模式获取存储适配器

    通过 APP_MODE 环境变量判断：
    - "client" (默认): 本地模式，使用 LocalStorageAdapter
    - "server": 云端模式，优先尝试 OSSStorageAdapter，失败则回退到 CloudStorageAdapter

    云端模式可通过 CACHE_TTL_SECONDS 环境变量配置缓存过期时间（默认 3600 秒）
    OSS 模式需要设置 CORONA_OSS_* 环境变量
    """
    global _adapter_instance

    if _adapter_instance is None:
        with _adapter_lock:
            if _adapter_instance is None:
                app_mode = os.environ.get("APP_MODE", "client").lower()

                if app_mode == "server":
                    # 云端模式：优先尝试 OSSStorageAdapter，失败则回退到 CloudStorageAdapter
                    ttl_str = os.environ.get("CACHE_TTL_SECONDS", "3600")
                    try:
                        ttl_seconds = int(ttl_str)
                    except ValueError:
                        ttl_seconds = MEMORY_CACHE_CONFIG["default_ttl_seconds"]

                    OSSStorageAdapter = _try_import_oss_adapter()
                    if OSSStorageAdapter is not None:
                        try:
                            _adapter_instance = OSSStorageAdapter(
                                ttl_seconds=ttl_seconds
                            )
                            logger.info(
                                f"存储适配器: OSSStorageAdapter (OSS 模式, TTL={ttl_seconds}s)"
                            )
                        except Exception as e:
                            logger.warning(
                                f"OSS 适配器初始化失败: {e}，回退到 CloudStorageAdapter"
                            )
                            _adapter_instance = CloudStorageAdapter(
                                ttl_seconds=ttl_seconds
                            )
                            logger.info(
                                f"存储适配器: CloudStorageAdapter (云端模式, TTL={ttl_seconds}s)"
                            )
                    else:
                        _adapter_instance = CloudStorageAdapter(ttl_seconds=ttl_seconds)
                        logger.info(
                            f"存储适配器: CloudStorageAdapter (云端模式, TTL={ttl_seconds}s)"
                        )

                else:
                    _adapter_instance = LocalStorageAdapter()
                    logger.info("存储适配器: LocalStorageAdapter (本地模式)")

    return _adapter_instance


def reset_storage_adapter() -> None:
    """重置存储适配器（用于测试）"""
    global _adapter_instance
    with _adapter_lock:
        _adapter_instance = None


def resolve_cache_url(
    cache_url: str, return_original_url: bool = False
) -> Optional[str]:
    """
    解析 cache:// URL（便捷函数）

    仅在云端模式下有效，本地模式返回原 URL。

    参数:
    - cache_url: cache://{cache_id} 格式的 URL
    - return_original_url: 是否返回原始云端 URL
      - False (默认): 返回 base64 data URI（安全，不泄露上游 URL）
      - True: 返回原始云端 URL（用于需要 HTTP URL 的上游 API）

    返回:
    - base64 data URI 或原始云端 URL
    """
    adapter = get_storage_adapter()
    if isinstance(adapter, CloudStorageAdapter):
        return adapter.resolve_cache_url(
            cache_url, return_original_url=return_original_url
        )
    return cache_url


__all__ = [
    "get_storage_adapter",
    "reset_storage_adapter",
    "resolve_cache_url",
]
