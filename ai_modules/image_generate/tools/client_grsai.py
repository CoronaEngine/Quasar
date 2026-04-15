from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ai_models.utils import BaseAPIClient
from ai_modules.providers.configs.dataclasses import ProviderConfig

logger = logging.getLogger(__name__)

_GRSAI_HTTP_CLIENT: Optional[httpx.Client] = None
_GRSAI_CLIENT_LOCK = threading.Lock()


def _get_grsai_http_client() -> httpx.Client:
    """获取全局共享的 GRSAI HTTP 客户端（线程安全单例）。"""
    global _GRSAI_HTTP_CLIENT
    if _GRSAI_HTTP_CLIENT is None:
        with _GRSAI_CLIENT_LOCK:
            if _GRSAI_HTTP_CLIENT is None:
                _GRSAI_HTTP_CLIENT = httpx.Client(
                    timeout=180.0,
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                    ),
                )
    return _GRSAI_HTTP_CLIENT


class GrsaiImageClient(BaseAPIClient):
    """GRSAI Nano-Banana 图像生成客户端（Legacy 模式）。"""

    def __init__(
        self,
        *,
        provider: ProviderConfig,
        model: str,
        base_url: str | None,
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
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": f"请帮我生成图像：{prompt}",
            "aspectRatio": resolution,
            "imageSize": image_size or "2K",
            "shutProgress": False,
        }

        urls = self._collect_http_image_urls(image_urls)
        if urls:
            payload["urls"] = urls

        headers = {
            "Content-Type": "application/json",
            **(self.headers or {}),
        }

        logger.debug(
            "GRSAI 图像生成请求: model=%s, has_urls=%s",
            self.model,
            bool(urls),
        )

        stream_deadline = time.time() + 180.0
        client = _get_grsai_http_client()

        with client.stream("POST", self.base_url, json=payload, headers=headers) as response:
            response.raise_for_status()

            data: Dict[str, Any] | None = None
            for line in response.iter_lines():
                if time.time() >= stream_deadline:
                    raise RuntimeError("图像生成超时: GRSAI 流式响应超时")

                if not line:
                    continue
                if isinstance(line, bytes):
                    try:
                        line = line.decode("utf-8")
                    except Exception:
                        continue

                if line.startswith("data:"):
                    line = line[5:].lstrip()

                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = chunk.get("status", "")
                if status == "running":
                    data = chunk
                    continue
                if status in ("succeeded", "failed"):
                    data = chunk
                    break

            if data is None:
                raise RuntimeError("未收到任何响应数据")

        return self._parse_response(data)

    @staticmethod
    def _collect_http_image_urls(image_urls: Optional[List[str]]) -> List[str]:
        """收集可用于 GRSAI 的 HTTP(S) 参考图 URL。"""
        if not image_urls:
            return []

        collected: List[str] = []
        for source in image_urls:
            if not source:
                continue

            # fileid:// 先解析到真实 URL，仅保留 HTTP(S)
            if source.startswith("fileid://"):
                from ai_media_resource import get_media_registry

                file_id = source[9:].lstrip("/")
                try:
                    resolved_url = get_media_registry().resolve(file_id, timeout=150.0)
                except Exception as e:
                    logger.warning("解析 fileid 失败: %s, error=%s", file_id, e)
                    continue

                if resolved_url.startswith(("http://", "https://")):
                    collected.append(resolved_url)
                else:
                    logger.warning("GRSAI 仅支持 HTTP 参考图，已忽略: %s", resolved_url)
                continue

            if source.startswith(("http://", "https://")):
                collected.append(source)
            else:
                logger.warning("GRSAI 仅支持 HTTP 参考图，已忽略: %s", source)

        return collected

    @staticmethod
    def _parse_response(body: Dict[str, Any]) -> Tuple[str, str]:
        status = body.get("status", "")
        if status == "failed":
            error_msg = body.get("failure_reason") or body.get("error") or "未知错误"
            raise RuntimeError(f"API 返回失败: {error_msg}")

        results = body.get("results") or []
        if not results:
            raise RuntimeError("API 返回的 results 为空")

        first_result = results[0]
        image_url = first_result.get("url")
        if not image_url:
            raise RuntimeError("API 返回的图像 URL 为空")

        return image_url, "image/png"


__all__ = ["GrsaiImageClient"]
