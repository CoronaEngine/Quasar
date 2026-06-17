from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ....ai_models.utils import BaseAPIClient
from ...providers.configs.dataclasses import ProviderConfig

logger = logging.getLogger(__name__)

_DMX_HTTP_CLIENT: Optional[httpx.Client] = None
_DMX_CLIENT_LOCK = threading.Lock()

# grsai aspectRatio → dmx size
_RESOLUTION_TO_SIZE: Dict[str, str] = {
    "1:1": "1024x1024",
    "16:9": "1536x1024",
    "9:16": "1024x1536",
    "4:3": "1024x1024",
    "3:4": "1024x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
}


def _get_dmx_http_client() -> httpx.Client:
    global _DMX_HTTP_CLIENT
    if _DMX_HTTP_CLIENT is None:
        with _DMX_CLIENT_LOCK:
            if _DMX_HTTP_CLIENT is None:
                _DMX_HTTP_CLIENT = httpx.Client(
                    timeout=300.0,
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                    ),
                )
    return _DMX_HTTP_CLIENT


def _resolve_image_size(resolution: str, image_size: Optional[str]) -> str:
    if image_size:
        size = image_size.strip()
        if "x" in size and size.split("x")[0].isdigit():
            return size
    return _RESOLUTION_TO_SIZE.get(resolution, "1024x1024")


class DmxImageClient(BaseAPIClient):
    """DMXAPI GPT Image 2 图像生成客户端（OpenAI 兼容协议）。"""

    def __init__(
        self,
        *,
        provider: ProviderConfig,
        model: str,
        base_url: str | None,
    ) -> None:
        super().__init__(provider, base_url)

        self.model = model or "gpt-image-2-ssvip"
        if not self.base_url:
            self.base_url = "https://www.dmxapi.cn/v1/images/generations"

    def generate(
        self,
        *,
        prompt: str,
        resolution: str,
        image_urls: Optional[List[str]] = None,
        image_size: Optional[str] = None,
    ) -> Tuple[str, str]:
        size = _resolve_image_size(resolution, image_size)

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "output_format": "png",
            "quality": "high",
            "moderation": "low",
        }

        headers = {
            "Content-Type": "application/json",
            **(self.headers or {}),
        }

        logger.debug(
            "DMX 图像生成请求: model=%s, size=%s",
            self.model,
            size,
        )

        client = _get_dmx_http_client()

        response = client.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=300.0,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"DMX 返回非 JSON 响应: {response.text[:500]}")

        return self._parse_response(data)

    @staticmethod
    def _parse_response(body: Dict[str, Any]) -> Tuple[str, str]:
        image_list = body.get("data") or []
        if not image_list:
            raise RuntimeError("DMX API 返回的 data 为空")

        first = image_list[0]

        if first.get("b64_json"):
            return f"data:image/png;base64,{first['b64_json']}", "image/png"

        image_url = first.get("url")
        if image_url:
            return image_url, "image/png"

        raise RuntimeError("DMX API 返回的图像既无 b64_json 也无 url")


__all__ = ["DmxImageClient"]
