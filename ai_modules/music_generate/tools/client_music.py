"""
Suno 音乐生成客户端
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ...providers.configs.dataclasses import ProviderConfig
from ....ai_models.utils import (
    BaseAPIClient,
    TaskPoller,
    retry_operation,
)


# 全局共享 HTTP 客户端连接池（线程安全）
_MUSIC_HTTP_CLIENT: Optional[httpx.Client] = None
_MUSIC_CLIENT_LOCK = threading.Lock()


def _get_music_http_client() -> httpx.Client:
    """获取全局共享的音乐生成 HTTP 客户端（线程安全单例）"""
    global _MUSIC_HTTP_CLIENT
    if _MUSIC_HTTP_CLIENT is None:
        with _MUSIC_CLIENT_LOCK:
            if _MUSIC_HTTP_CLIENT is None:
                _MUSIC_HTTP_CLIENT = httpx.Client(
                    timeout=60.0,
                    limits=httpx.Limits(
                        max_connections=20, max_keepalive_connections=10
                    ),
                )
    return _MUSIC_HTTP_CLIENT


class SunoMusicClient(BaseAPIClient):
    """Suno 音乐生成客户端"""

    def __init__(
        self,
        provider: ProviderConfig,
        base_url: str | None = None,
    ) -> None:
        super().__init__(provider, base_url)
        if not self.base_url:
            self.base_url = "https://api.sunoapi.org"

    @retry_operation(max_retries=3)
    def generate_music(
        self,
        prompt: str,
        style: Optional[str] = None,
        model: str = "V5",
        instrumental: bool = True,
        wait: bool = True,
        max_wait_seconds: int = 150,
        poll_interval: float = 3.0,
    ) -> Dict[str, Any]:
        """
        生成音乐

        Args:
            prompt: 提示词
            style: 风格
            model: 模型版本
            instrumental: 是否纯音乐
            wait: 是否等待完成
            max_wait_seconds: 最大等待时间
            poll_interval: 轮询间隔

        Returns:
            包含任务ID或生成结果的字典
        """
        payload = {
            "prompt": prompt,
            "model": model,
            "customMode": bool(style),
            "instrumental": instrumental,
            "callBackUrl": "https://example.com/callback",  # 占位符
        }

        if style:
            payload["style"] = style
            payload["title"] = prompt[:80]

        initial = self._post_generate(payload)

        # 提取任务ID
        data_obj = initial.get("data", {})
        task_id = data_obj.get("taskId") or initial.get("taskId")

        if not task_id:
            raise RuntimeError(f"API未返回任务ID: {initial}")

        if not wait:
            return {"task_id": task_id, "status": "PENDING"}

        # 轮询任务
        poller = TaskPoller(interval=poll_interval, timeout=max_wait_seconds)

        def check_status(tid: str) -> Tuple[str, Any, Optional[str]]:
            details = self._get_details(tid)
            data = details.get("data", {})
            status = data.get("status")

            # 映射状态
            # Suno API status: PENDING, TEXT_SUCCESS, FIRST_SUCCESS, SUCCESS,
            # CREATE_TASK_FAILED, GENERATE_AUDIO_FAILED, CALLBACK_EXCEPTION, SENSITIVE_WORD_ERROR

            if status == "SUCCESS":
                return "SUCCEEDED", self._parse_result(data), None

            if status in (
                "CREATE_TASK_FAILED",
                "GENERATE_AUDIO_FAILED",
                "CALLBACK_EXCEPTION",
                "SENSITIVE_WORD_ERROR",
                "ERROR",
            ):
                error_msg = data.get("errorMessage") or f"Status: {status}"
                return "FAILED", None, error_msg

            return "RUNNING", None, None

        return poller.poll(task_id, check_status)

    def _post_generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送生成请求"""
        url = f"{self.base_url}/api/v1/generate"
        client = _get_music_http_client()
        resp = client.post(url, json=payload, headers=self.headers, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 200:
            raise RuntimeError(
                f"API返回错误: code={result.get('code')}, msg={result.get('msg')}"
            )
        return result

    def _get_details(self, task_id: str) -> Dict[str, Any]:
        """获取任务详情"""
        url = f"{self.base_url}/api/v1/generate/record-info"
        params = {"taskId": task_id}
        client = _get_music_http_client()
        resp = client.get(url, params=params, headers=self.headers, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 200:
            raise RuntimeError(
                f"查询失败: code={result.get('code')}, msg={result.get('msg')}"
            )
        return result

    def _parse_result(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析成功结果"""
        response_obj = data.get("response", {})
        suno_data = response_obj.get("sunoData", [])

        results = []
        if isinstance(suno_data, list):
            for item in suno_data:
                if item.get("audioUrl"):
                    results.append(
                        {
                            "audio_url": item.get("audioUrl"),
                            "title": item.get("title"),
                            "duration": item.get("duration"),
                            "image_url": item.get("imageUrl"),
                        }
                    )
        return results
