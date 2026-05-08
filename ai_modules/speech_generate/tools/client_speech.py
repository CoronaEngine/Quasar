"""
火山引擎语音合成服务客户端
支持 HTTP 异步调用方式 (v3 API)
"""

import threading
import uuid
from typing import Optional, Dict, Any, Tuple
import httpx

from ....ai_models.utils import TaskPoller
from ..configs.dataclasses import SpeechAppConfig, SpeechAudioConfig

# 全局共享 HTTP 客户端连接池（线程安全）
_TTS_HTTP_CLIENT: Optional[httpx.Client] = None
_TTS_CLIENT_LOCK = threading.Lock()


def _get_tts_http_client() -> httpx.Client:
    """获取全局共享的 TTS HTTP 客户端（线程安全单例）"""
    global _TTS_HTTP_CLIENT
    if _TTS_HTTP_CLIENT is None:
        with _TTS_CLIENT_LOCK:
            if _TTS_HTTP_CLIENT is None:
                _TTS_HTTP_CLIENT = httpx.Client(
                    timeout=30.0,
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                )
    return _TTS_HTTP_CLIENT


class TTSClient:
    """语音合成服务客户端"""

    SUBMIT_API = "https://openspeech.bytedance.com/api/v3/tts/submit"
    QUERY_API = "https://openspeech.bytedance.com/api/v3/tts/query"

    def __init__(self, app_config: SpeechAppConfig):
        self.app_config = app_config
        self._client = _get_tts_http_client()

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "X-Api-App-Id": self.app_config.appid,
            "X-Api-Access-Key": self.app_config.token,
            "X-Api-Resource-Id": "volc.service_type.10029",
            "X-Api-Request-Id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }

    def _submit(self, text: str, audio_config: SpeechAudioConfig) -> str:
        """提交语音合成任务，返回 task_id"""
        # 构建请求体
        speech_rate = int((audio_config.speed_ratio - 1.0) * 100)
        loudness_rate = int((audio_config.loudness_ratio - 1.0) * 100)

        body: Dict[str, Any] = {
            "user": {"uid": self.app_config.uid},
            "req_params": {
                "text": text,
                "speaker": audio_config.voice_type,
                "audio_params": {
                    "format": audio_config.encoding,
                    "sample_rate": audio_config.rate,
                    "speech_rate": speech_rate,
                    "loudness_rate": loudness_rate,
                },
            },
        }

        # 添加情感参数
        if audio_config.emotion:
            body["req_params"]["audio_params"]["emotion"] = audio_config.emotion
            if audio_config.emotion_scale:
                body["req_params"]["audio_params"]["emotion_scale"] = audio_config.emotion_scale

        # 发送请求
        resp = self._client.post(self.SUBMIT_API, headers=self._build_headers(), json=body)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 20000000:
            raise RuntimeError(f"TTS提交失败: [{result.get('code')}] {result.get('message')}")

        task_id = result.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError("API未返回task_id")

        return task_id

    def _query(self, task_id: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        """
        查询任务状态，返回 TaskPoller 期望的格式: (status, result, error_msg)
        status: "PROCESSING" / "SUCCEEDED" / "FAILED"
        """
        resp = self._client.post(
            self.QUERY_API,
            headers=self._build_headers(),
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 20000000:
            return "FAILED", None, result.get("message", "查询失败")

        data = result.get("data", {})
        task_status = data.get("task_status")

        # task_status: 1=Running, 2=Success, 3=Failure
        if task_status == 2:
            return "SUCCEEDED", {
                "task_id": task_id,
                "audio_url": data.get("audio_url"),
                "url_expire_time": data.get("url_expire_time"),
                "duration": data.get("synthesize_text_length"),
                "sentences": data.get("sentences"),
            }, None
        elif task_status == 1:
            return "PROCESSING", None, None
        elif task_status == 3:
            return "FAILED", None, result.get("message", "任务失败")
        else:
            return "FAILED", None, f"未知状态: {task_status}"

    def synthesize(
        self,
        text: str,
        audio_config: SpeechAudioConfig,
        max_wait_seconds: int = 60,
        poll_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """
        语音合成（提交任务并轮询直到完成）

        Args:
            text: 待合成文本
            audio_config: 音频配置
            max_wait_seconds: 最大等待时间（秒）
            poll_interval: 轮询间隔（秒）

        Returns:
            包含 audio_url 等信息的字典
        """
        task_id = self._submit(text, audio_config)
        poller = TaskPoller(interval=poll_interval, timeout=max_wait_seconds)
        return poller.poll(task_id, self._query)
