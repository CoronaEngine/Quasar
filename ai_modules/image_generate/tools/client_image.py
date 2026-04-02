from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ai_modules.providers.configs.dataclasses import ProviderConfig

# from config.ai_config import ProviderConfig
from ai_models.utils import (
    retry_operation,
    BaseAPIClient,
    file_url_to_data_uri,
)

import logging

logger = logging.getLogger(__name__)

# 全局共享 HTTP 客户端连接池（线程安全）
_IMAGE_HTTP_CLIENT: Optional[httpx.Client] = None
_IMAGE_CLIENT_LOCK = threading.Lock()


def _get_image_http_client() -> httpx.Client:
    """获取全局共享的图像生成 HTTP 客户端（线程安全单例）"""
    global _IMAGE_HTTP_CLIENT
    if _IMAGE_HTTP_CLIENT is None:
        with _IMAGE_CLIENT_LOCK:
            if _IMAGE_HTTP_CLIENT is None:
                _IMAGE_HTTP_CLIENT = httpx.Client(
                    timeout=150.0,
                    limits=httpx.Limits(
                        max_connections=20, max_keepalive_connections=10
                    ),
                )
    return _IMAGE_HTTP_CLIENT


class LingyaImageClient(BaseAPIClient):
    """负责与灵芽图片生成/编辑服务交互的客户端。

    根据是否提供参考图片自动选择纯文本生成或编辑接口。
    """

    def __init__(
        self, *, provider: ProviderConfig, model: str, base_url: str | None
    ) -> None:
        super().__init__(provider, base_url)

        self.model = model

        if not self.base_url:
            raise RuntimeError(f"Provider '{provider.name}' 缺少 base_url。")

    def generate(
        self,
        *,
        prompt: str,
        resolution: str,
        image_urls: Optional[List[str]] = None,
        image_size: Optional[str] = None,
    ) -> Tuple[str, str]:
        images_data = self._collect_image_data(image_urls)

        logger.info(
            "图像生成配置："
            f"基础 URL: {self.base_url}, api_key: {self.api_key}"
        )

        return self._generate_request(
            prompt=prompt,
            resolution=resolution,
            images=images_data or None,
            image_size=image_size,
        )

    @retry_operation(max_retries=1)
    def _generate_request(
        self,
        *,
        prompt: str,
        resolution: Optional[str] = None,
        images: Optional[List[str]] = None,
        image_size: Optional[str] = None,
    ) -> Tuple[str, str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "response_format": "url",
        }

        if images:
            payload["image"] = images
        else:
            if not resolution:
                raise ValueError(
                    "resolution 不能为空（纯文本生成模式需要 aspect_ratio）。"
                )
            payload["aspect_ratio"] = resolution
            if self.model == "nano-banana-pro" and image_size:
                payload["image_size"] = image_size

        client = _get_image_http_client()
        response = client.post(
            self.base_url, json=payload, headers=self.headers, timeout=150
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _parse_response(self, body: Dict[str, Any]) -> Tuple[str, str]:
        """解析API响应，返回 (image_url, mime_type)。

        根据API文档，当指定response_format="url"时，API保证返回URL字段。
        """
        images = body.get("data") or []
        if not images:
            raise RuntimeError("Lingya 服务未返回图像数据。")
        item = images[0]

        # API应该返回URL字段（因为我们指定了response_format="url"）
        if "url" in item:
            mime = item.get("mime_type") or "image/png"
            return item["url"], mime

        # 不应该走到这里，如果走到这里说明API行为异常
        raise RuntimeError(
            "API未按预期返回URL字段。请检查response_format参数是否生效。"
        )

    @staticmethod
    def _collect_image_data(image_urls: Optional[List[str]]) -> List[str]:
        """收集图片数据，支持URL或本地路径，返回可用于API的图片数据列表"""
        if not image_urls:
            return []
        images: List[str] = []
        for source in image_urls:
            if not source:
                continue
            # 如果是 fileid:// URL，解析为真实 URL
            if source.startswith("fileid://"):
                from ai_media_resource import get_media_registry

                file_id = source[9:].lstrip("/")
                try:
                    resolved_url = get_media_registry().resolve(file_id, timeout=150.0)
                    # 递归处理解析后的 URL（可能是 file:// 或 http://）
                    if resolved_url.startswith(("http://", "https://")):
                        images.append(resolved_url)
                    elif resolved_url.startswith("data:"):
                        images.append(resolved_url)
                    else:
                        data = file_url_to_data_uri(resolved_url)
                        if data:
                            images.append(data)
                except Exception as e:
                    # 记录错误但不中断整个流程
                    import logging

                    logging.getLogger(__name__).error(
                        f"解析 file_id {file_id} 失败: {e}"
                    )
                    continue
            # 如果是HTTP(S) URL，直接使用
            elif source.startswith(("http://", "https://")):
                images.append(source)
            # 如果是data URI，直接使用
            elif source.startswith("data:"):
                images.append(source)
            # 如果是本地路径，转换为base64
            else:
                data = file_url_to_data_uri(source)
                if data:
                    images.append(data)
        return images


__all__ = ["LingyaImageClient"]
