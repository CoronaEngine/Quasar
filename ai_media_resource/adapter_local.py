"""
本地存储适配器

下载资源到本地文件系统，返回 file:// URL。
"""

from __future__ import annotations

import base64
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import requests

from ai_media_resource.adapter_base import (
    StorageAdapter,
    normalize_to_data_uri,
)
from ai_media_resource.result import StorageResult

logger = logging.getLogger(__name__)


class LocalStorageAdapter(StorageAdapter):
    """
    本地存储适配器

    下载资源到本地，返回 file:// URL。
    桥接 Backend.local_storage.MediaStore。
    """

    def __init__(self):
        pass

    @property
    def save_path(self) -> Path:
        """每次访问时动态解析存储路径，确保跟随当前活跃项目。"""
        return self._resolve_save_path()

    def _resolve_save_path(self) -> Path:
        """解析本地存储路径：优先使用项目路径下的 media/ 目录，未配置时自动推算"""
        try:
            from config.paths_config import get_project_media_dir
            return get_project_media_dir()
        except Exception:
            pass
        fallback = Path(__file__).parent.parent / "local_storage"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def save_from_url(
        self,
        cloud_url: str,
        session_id: str,
        resource_type: str,
        original_name: Optional[str] = None,
        url_expire_time: Optional[int] = None,
    ) -> StorageResult:
        """下载资源到本地，返回 file:// URL"""
        _id = uuid.uuid1()
        if original_name:
            # 清理文件名，移除非法字符
            safe_name = re.sub(r'[^\w\-_.]', '_', original_name)
            name, ext = os.path.splitext(safe_name)

            # 如果没有扩展名，根据资源类型添加
            if not ext and resource_type:
                if resource_type.startswith('image'):
                    ext = '.jpg' if 'jpeg' in resource_type else '.png'
                elif resource_type.startswith('video'):
                    ext = '.mp4'
                elif resource_type.startswith('audio'):
                    ext = '.mp3'
                else:
                    ext = '.bin'

            filename = f"{name}_{session_id}_{_id}{ext}"
        else:
            # 生成随机文件名
            if resource_type.startswith('image'):
                ext = '.jpg' if 'jpeg' in resource_type else '.png'
            elif resource_type.startswith('video'):
                ext = '.mp4'
            elif resource_type.startswith('audio'):
                ext = '.mp3'
            else:
                ext = '.bin'

            filename = f"resource_{session_id}_{_id}{ext}"

        local_path = self.save_path / filename

        # 3. 下载文件
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(cloud_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_url = f"{local_path}"
        logger.info(f"资源已下载到本地: {cloud_url} -> {file_url}")
        return StorageResult(url=str(file_url), url_expire_time=None)

    def save_from_base64(
        self,
        data_uri: str,
        session_id: str,
        resource_type: str,
        filename_prefix: str = "resource",
        url_expire_time: Optional[int] = None,
    ) -> StorageResult:
        """将 base64 数据保存到本地，返回 file:// URL"""
        normalized_data = normalize_to_data_uri(data_uri, resource_type)

        header, data = normalized_data.split(',', 1)

        # 提取MIME类型
        mime_match = re.match(r'data:(.*?);base64', header)
        if not mime_match:
            raise ValueError("无法解析data URI的MIME类型")

        mime_type = mime_match.group(1)

        # 3. 生成文件名
        if resource_type:
            # 根据资源类型确定扩展名
            if resource_type.startswith('image'):
                if 'jpeg' in resource_type or 'jpg' in resource_type:
                    ext = '.jpg'
                elif 'png' in resource_type:
                    ext = '.png'
                elif 'gif' in resource_type:
                    ext = '.gif'
                else:
                    ext = '.jpg'
            elif resource_type.startswith('video'):
                ext = '.mp4'
            elif resource_type.startswith('audio'):
                ext = '.mp3'
            elif resource_type.startswith('application/pdf'):
                ext = '.pdf'
            else:
                ext = '.bin'
        else:
            # 从MIME类型推断扩展名
            if 'jpeg' in mime_type or 'jpg' in mime_type:
                ext = '.jpg'
            elif 'png' in mime_type:
                ext = '.png'
            elif 'gif' in mime_type:
                ext = '.gif'
            elif 'pdf' in mime_type:
                ext = '.pdf'
            else:
                ext = '.bin'

        _id = uuid.uuid1()
        filename = f"{filename_prefix}_{_id}{ext}"
        # 4. 构建本地路径
        local_path = self.save_path / filename

        # 5. 解码并保存base64数据
        try:
            # 解码base64数据
            binary_data = base64.b64decode(data)

            # 保存文件
            with open(local_path, 'wb') as f:
                f.write(binary_data)

        except base64.binascii.Error:
            raise ValueError("Base64数据解码失败")

        # 6. 返回结果
        file_url = f"{local_path}"

        logger.info(f"Base64 数据已保存到本地: {file_url}")
        return StorageResult(url=file_url, url_expire_time=None)


__all__ = ["LocalStorageAdapter"]
