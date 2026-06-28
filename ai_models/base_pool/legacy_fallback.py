"""
旧客户端降级适配器

当私密仓库的账号池系统不可用时，使用旧客户端的单例模式作为降级方案。
这些适配器包装现有的 client_*.py 客户端，提供与账号池一致的接口。
"""

from __future__ import annotations

import logging
import threading
import base64
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import httpx

from .category import MediaCategory
from .requests import (
    MediaRequest,
    ImageRequest,
    VideoRequest,
    MusicRequest,
    SpeechRequest,
    ChatRequest,
    OmniRequest,
    DetectionRequest,
)
from .responses import (
    MediaResult,
    MultiMediaResult,
    ChatResult,
)
from ...ai_modules.providers.configs.dataclasses import ProviderConfig
# from ...ai_modules.image_generate.tools.client_grsai import GrsaiImageClient  # grsai 已弃用
from ...ai_modules.image_generate.tools.client_dmx import DmxImageClient
from ...ai_modules.speech_generate.configs.dataclasses import SpeechAudioConfig, SpeechAppConfig

logger = logging.getLogger(__name__)

# ============================================================================
# 单例客户端缓存
# ============================================================================

_legacy_clients: Dict[MediaCategory, Any] = {}
_legacy_clients_lock = threading.Lock()


# ============================================================================
# 可扩展图像客户端工厂注册
# ============================================================================


_IMAGE_CLIENT_CLASSES: Dict[str, type] = {
    # provider 名称与 image.provider 对齐
    # "grsai_image": GrsaiImageClient,  # grsai 已弃用，改用 dmx
    "dmx_image": DmxImageClient,
}


def _get_legacy_image_client():
    """获取旧图像客户端单例"""
    from ...ai_config.ai_config import get_ai_config

    config = get_ai_config()
    image_cfg = config.image

    if not image_cfg.enable or not image_cfg.provider:
        return None

    provider = config.providers.get(image_cfg.provider)
    if not provider or not provider.api_key:
        return None

    key = (image_cfg.provider or "").strip().lower()
    client_cls = _IMAGE_CLIENT_CLASSES.get(key)
    if client_cls is None:
        logger.warning("Unsupported legacy image provider: %s", key)
        return None

    return client_cls(
        provider=provider,
        model=image_cfg.model,
        base_url=image_cfg.base_url,
    )


def _get_legacy_video_client():
    """获取旧视频客户端单例"""
    from ...ai_config.ai_config import get_ai_config
    from ...ai_modules.video_generate.tools.client_video import DashScopeVideoClient

    config = get_ai_config()
    video_cfg = config.video

    if not video_cfg.enable or not video_cfg.provider:
        return None

    provider = config.providers.get(video_cfg.provider)
    if not provider or not provider.api_key:
        return None

    return DashScopeVideoClient(
        provider=provider,
        model=video_cfg.model,
        base_url=video_cfg.base_url,
    )


def _get_legacy_speech_client():
    """获取旧语音合成客户端单例"""
    from ...ai_config.ai_config import (
        get_ai_config,
        # SpeechAppConfig,
    )
    from ...ai_modules.speech_generate.tools.client_speech import TTSClient

    config = get_ai_config()

    # TTS 使用独立配置
    if not config.tts.appid or not config.tts.token:
        return None

    app_config = SpeechAppConfig(
        appid=config.tts.appid,
        token=config.tts.token,
    )

    return TTSClient(app_config=app_config)


def _get_legacy_music_client():
    """获取旧音乐生成客户端单例"""
    from ...ai_config.ai_config import (
        get_ai_config,
        # ProviderConfig,
    )
    from ...ai_modules.music_generate.tools.client_music import SunoMusicClient

    config = get_ai_config()

    # 音乐使用独立配置
    if not config.music.api_key:
        return None

    # 构造一个伪 provider
    provider = ProviderConfig(
        name="suno",
        type="suno",
        api_key=config.music.api_key,
        base_url=config.music.base_url,
    )

    return SunoMusicClient(provider=provider, base_url=config.music.base_url)


def _get_legacy_chat_client():
    """
    获取旧聊天客户端单例

    返回 LangChain BaseChatModel 实例，用于 Agent 和工具调用。
    """
    from ...ai_config.ai_config import get_ai_config
    from ...ai_modules.text_generate.tools.chat_loader import get_chat_model

    config = get_ai_config()
    chat_cfg = config.chat

    # 检查 provider 是否可用
    if chat_cfg.provider not in config.providers:
        return None

    provider = config.providers[chat_cfg.provider]
    if not provider.api_key:
        return None

    return get_chat_model(
        config,
        provider_name=chat_cfg.provider,
        model_name=chat_cfg.model,
        temperature=chat_cfg.temperature,
        request_timeout=chat_cfg.request_timeout,
    )


def _get_legacy_omni_client():
    """
    获取旧多模态理解客户端单例

    返回 (OpenAI 客户端, OmniModelConfig) 元组。
    """
    from ...ai_config.ai_config import get_ai_config
    from openai import OpenAI

    config = get_ai_config()
    omni_cfg = config.omni

    if not omni_cfg.enable or not omni_cfg.provider or not omni_cfg.model:
        return None

    if omni_cfg.provider not in config.providers:
        return None

    provider = config.providers[omni_cfg.provider]
    if not provider.api_key or not provider.base_url:
        return None

    client = OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=omni_cfg.request_timeout,
        max_retries=0,
    )

    return (client, omni_cfg)


def _get_legacy_detection_client():
    """
    获取旧目标检测客户端单例

    返回 (OpenAI 客户端, DetectionModelConfig) 元组。
    """
    from ...ai_config.ai_config import get_ai_config
    from openai import OpenAI

    config = get_ai_config()
    detection_cfg = config.detection

    if not detection_cfg.enable or not detection_cfg.provider or not detection_cfg.model:
        return None

    if detection_cfg.provider not in config.providers:
        return None

    provider = config.providers[detection_cfg.provider]
    if not provider.api_key or not provider.base_url:
        return None

    client = OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=detection_cfg.request_timeout,
        max_retries=0,
    )

    return (client, detection_cfg)


def get_legacy_client(category: MediaCategory) -> Optional[Any]:
    """
    获取指定类别的旧客户端单例

    线程安全，使用双重检查锁定。
    """
    if category in _legacy_clients:
        return _legacy_clients[category]

    with _legacy_clients_lock:
        if category in _legacy_clients:
            return _legacy_clients[category]

        # 按类别创建客户端
        client = None
        try:
            if category == MediaCategory.IMAGE:
                client = _get_legacy_image_client()
            elif category == MediaCategory.VIDEO:
                client = _get_legacy_video_client()
            elif category == MediaCategory.SPEECH:
                client = _get_legacy_speech_client()
            elif category == MediaCategory.MUSIC:
                client = _get_legacy_music_client()
            elif category == MediaCategory.AGENT:
                # Agent 推理用 LLM
                client = _get_legacy_chat_client()
            elif category == MediaCategory.TEXT:
                # 文案工具用 LLM（降级时也用同一个客户端）
                client = _get_legacy_chat_client()
            elif category == MediaCategory.OMNI:
                # 多模态理解 VLM
                client = _get_legacy_omni_client()
            elif category == MediaCategory.DETECTION:
                # 目标检测 VLM
                client = _get_legacy_detection_client()
            else:
                logger.warning(f"不支持的降级类别: {category}")
        except Exception as e:
            logger.error(f"创建旧客户端失败 [{category}]: {e}")

        _legacy_clients[category] = client
        return client


def clear_legacy_clients() -> None:
    """清除旧客户端缓存（用于测试）"""
    global _legacy_clients
    with _legacy_clients_lock:
        _legacy_clients.clear()


# ============================================================================
# 降级任务创建
# ============================================================================


def create_legacy_image_task(
    request: ImageRequest,
) -> Optional[Callable[[], MediaResult]]:
    """
    创建旧图像客户端任务
    """
    client = get_legacy_client(MediaCategory.IMAGE)
    if client is None:
        return None

    def task() -> MediaResult:
        url, mime_type = client.generate(
            prompt=request.prompt,
            resolution=request.resolution,
            image_urls=request.image_urls if request.image_urls else None,
            image_size=request.image_size,
        )
        return MediaResult(
            url=url,
            mime_type=mime_type,
            metadata={"resolution": request.resolution, "image_size": request.image_size},
        )

    return task


def create_legacy_video_task(
    request: VideoRequest,
) -> Optional[Callable[[], MediaResult]]:
    """
    创建旧视频客户端任务
    """
    client = get_legacy_client(MediaCategory.VIDEO)
    if client is None:
        return None

    def task() -> MediaResult:
        result = client.generate_video_from_image(
            prompt=request.prompt,
            image_url=request.image_url,
            resolution=request.resolution,
            prompt_extend=request.prompt_extend,
        )

        output = result.get("output", {})
        video_url = output.get("video_url", "")

        return MediaResult(
            url=video_url,
            mime_type="video/mp4",
            metadata={
                "resolution": request.resolution,
                "actual_prompt": output.get("actual_prompt"),
            },
        )

    return task


def create_legacy_speech_task(
    request: SpeechRequest,
) -> Optional[Callable[[], MediaResult]]:
    """
    创建旧语音合成客户端任务
    """
    client = get_legacy_client(MediaCategory.SPEECH)
    if client is None:
        return None

    # from config.ai_config import SpeechAudioConfig

    def task() -> MediaResult:
        audio_config = SpeechAudioConfig(
            voice_type=request.voice_type,
            speed_ratio=request.speed_ratio,
            loudness_ratio=request.loudness_ratio,
            encoding=request.encoding,
            rate=request.sample_rate,
        )

        result = client.synthesize(
            text=request.text,
            audio_config=audio_config,
        )

        return MediaResult(
            url=result.get("audio_url", ""),
            mime_type=f"audio/{request.encoding}",
            url_expire_time=result.get("url_expire_time"),
            metadata={
                "duration": result.get("duration"),
            },
        )

    return task


def create_legacy_music_task(
    request: MusicRequest,
) -> Optional[Callable[[], Union[MediaResult, MultiMediaResult]]]:
    """
    创建旧音乐生成客户端任务
    """
    client = get_legacy_client(MediaCategory.MUSIC)
    if client is None:
        return None

    def task() -> Union[MediaResult, MultiMediaResult]:
        result = client.generate_music(
            prompt=request.prompt,
            style=request.style,
            model=request.model,
            wait=True,
        )

        # 解析结果（可能是多首）
        if isinstance(result, list) and len(result) > 0:
            primary = MediaResult(
                url=result[0].get("audio_url", ""),
                mime_type="audio/mp3",
                metadata={
                    "title": result[0].get("title"),
                    "duration": result[0].get("duration"),
                    "image_url": result[0].get("image_url"),
                },
            )

            extras = [
                MediaResult(
                    url=r.get("audio_url", ""),
                    mime_type="audio/mp3",
                    metadata={
                        "title": r.get("title"),
                        "duration": r.get("duration"),
                        "image_url": r.get("image_url"),
                    },
                )
                for r in result[1:]
            ]

            if extras:
                return MultiMediaResult(primary=primary, extras=extras)
            return primary

        # 单个结果
        return MediaResult(
            url=result.get("audio_url", "") if isinstance(result, dict) else "",
            mime_type="audio/mp3",
            metadata=result if isinstance(result, dict) else {},
        )

    return task


def create_legacy_chat_task(
    request: ChatRequest,
    category: MediaCategory = MediaCategory.AGENT,
) -> Optional[Callable[[], ChatResult]]:
    """
    创建旧聊天客户端任务

    注意：Chat 任务与媒体任务不同，返回 ChatResult 而非 MediaResult。

    参数:
    - request: ChatRequest
    - category: MediaCategory.AGENT 或 MediaCategory.TEXT
    """
    client = get_legacy_client(category)
    if client is None:
        return None

    def task() -> ChatResult:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        # 将 ChatRequest.messages 转换为 LangChain 格式
        messages = []
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        # 调用 LLM
        response = client.invoke(messages)
        content = response.content or ""

        # 处理数组类型的 content
        if isinstance(content, list):
            content = "\n".join(
                [b["text"] for b in content if b.get("type") == "text"]
            )

        return ChatResult(
            content=content,
            usage=response.response_metadata.get("token_usage"),
            metadata={"model": response.response_metadata.get("model_name")},
        )

    return task


def create_legacy_omni_task(
    request: OmniRequest,
) -> Optional[Callable[[], MediaResult]]:
    """
    创建旧多模态理解客户端任务

    返回 MediaResult，content 为 VLM 分析的文本结果。
    """
    client_tuple = get_legacy_client(MediaCategory.OMNI)
    if client_tuple is None:
        return None

    client, omni_config = client_tuple

    def task() -> MediaResult:
        from ...ai_tools.response_adapter import FILEID_SCHEME
        from ...ai_media_resource import get_media_registry
        from ..utils import file_url_to_data_uri

        # 解析 URL 的辅助函数
        def resolve_url(url: str) -> str:
            normalized = (url or "").strip()
            if not normalized:
                return ""

            if url.startswith(FILEID_SCHEME):
                file_id = url[len(FILEID_SCHEME):]
                local_candidate = Path(file_id)
                if local_candidate.exists():
                    return file_url_to_data_uri(local_candidate.resolve().as_uri())
                registry = get_media_registry()
                result = registry.resolve(file_id)
                if isinstance(result, dict):
                    normalized = str(result.get("url", ""))
                else:
                    normalized = str(result)

            # OpenAI-compatible VLM 侧通常无法访问本地路径，统一转换为 data URI。
            if normalized.startswith("file://"):
                return file_url_to_data_uri(normalized)

            local_path = Path(normalized)
            if local_path.exists():
                return file_url_to_data_uri(local_path.resolve().as_uri())

            # 供应商侧常无法直接访问临时/私有 URL，这里主动下载并内联成 data URI。
            if normalized.startswith(("http://", "https://")):
                try:
                    with httpx.Client(timeout=30.0, follow_redirects=True) as c:
                        r = c.get(normalized)
                        r.raise_for_status()
                        mime = r.headers.get("content-type", "").split(";")[0].strip().lower()
                        if not mime.startswith("image/"):
                            raise ValueError(f"URL 不是图片资源: content-type={mime}")
                        b64 = base64.b64encode(r.content).decode("utf-8")
                        return f"data:{mime};base64,{b64}"
                except Exception as e:
                    raise ValueError(
                        f"Omni 无法下载图片 URL: {normalized[:120]}, err={e}"
                    ) from e

            if normalized.startswith(("http://", "https://", "data:")):
                return normalized

            raise ValueError(
                f"Omni 不支持的图片输入，无法访问: {normalized[:120]}"
            )

            return normalized

        # 解析所有 URL
        resolved_image_urls = [resolve_url(url) for url in request.image_urls]
        resolved_video_urls = [resolve_url(url) for url in request.video_urls]
        resolved_audio_urls = [resolve_url(url) for url in request.audio_urls]

        if resolved_image_urls:
            scheme_stats = {"data": 0, "http": 0, "other": 0}
            for u in resolved_image_urls:
                if u.startswith("data:"):
                    scheme_stats["data"] += 1
                elif u.startswith(("http://", "https://")):
                    scheme_stats["http"] += 1
                else:
                    scheme_stats["other"] += 1
            logger.info(
                "[Omni] image url normalized: total=%s, data=%s, http=%s, other=%s",
                len(resolved_image_urls),
                scheme_stats["data"],
                scheme_stats["http"],
                scheme_stats["other"],
            )

        # 构建多模态消息 content
        content = []

        for url in resolved_image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": omni_config.image_detail},
            })

        for url in resolved_video_urls:
            content.append({
                "type": "video_url",
                "video_url": {
                    "url": url,
                    "detail": omni_config.image_detail,
                    "max_frames": omni_config.max_frames,
                    "fps": omni_config.fps,
                },
            })

        for url in resolved_audio_urls:
            content.append({"type": "audio_url", "audio_url": {"url": url}})

        content.append({"type": "text", "text": request.prompt})

        # 调用 VLM API
        response = client.chat.completions.create(
            model=omni_config.model,
            messages=[{"role": "user", "content": content}],
        )

        vlm_description = response.choices[0].message.content

        return MediaResult(
            url="",  # VLM 分析不产生 URL
            mime_type="text/plain",
            metadata={
                "analysis_result": vlm_description,
                "model": omni_config.model,
            },
        )

    return task


def create_legacy_detection_task(
    request: DetectionRequest,
) -> Optional[Callable[[], MediaResult]]:
    """
    创建旧目标检测客户端任务

    返回 MediaResult，metadata 中包含检测结果列表。
    注意：此函数在公共项目中提供基本框架，完整实现在私有工具中。
    """
    client_tuple = get_legacy_client(MediaCategory.DETECTION)
    if client_tuple is None:
        return None

    client, detection_config = client_tuple

    def task() -> MediaResult:
        from ...ai_tools.response_adapter import FILEID_SCHEME
        from ...ai_media_resource import get_media_registry
        from ..utils import file_url_to_data_uri

        # 解析 URL
        image_url = request.image_url
        if image_url.startswith(FILEID_SCHEME):
            file_id = image_url[len(FILEID_SCHEME):]
            registry = get_media_registry()
            image_url = registry.resolve(file_id)
            if image_url.startswith("file://"):
                image_url = file_url_to_data_uri(image_url)
        elif image_url.startswith("file://"):
            image_url = file_url_to_data_uri(image_url)

        # 构建检测提示词（简化版本，完整版本在私有工具中）
        prompt = "请检测图片中的所有对象，返回JSON数组格式，每个对象包含label、box（归一化坐标[x_min,y_min,x_max,y_max]）和description字段。"
        if request.target_description:
            prompt = f"重点检测：{request.target_description}。" + prompt

        # 构建多模态消息
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": detection_config.image_detail,
                },
            },
            {"type": "text", "text": prompt},
        ]

        # 调用 VLM API
        response = client.chat.completions.create(
            model=detection_config.model,
            messages=[{"role": "user", "content": content}],
        )

        vlm_response = response.choices[0].message.content

        return MediaResult(
            url="",
            mime_type="application/json",
            metadata={
                "detection_result": vlm_response,
                "model": detection_config.model,
            },
        )

    return task


def create_legacy_task(
    category: MediaCategory,
    request: MediaRequest,
) -> Optional[Callable[[], Union[MediaResult, MultiMediaResult, ChatResult]]]:
    """
    根据类别创建旧客户端任务

    这是降级模式的统一入口。
    """
    if category == MediaCategory.IMAGE and isinstance(request, ImageRequest):
        return create_legacy_image_task(request)
    elif category == MediaCategory.VIDEO and isinstance(request, VideoRequest):
        return create_legacy_video_task(request)
    elif category == MediaCategory.SPEECH and isinstance(request, SpeechRequest):
        return create_legacy_speech_task(request)
    elif category == MediaCategory.MUSIC and isinstance(request, MusicRequest):
        return create_legacy_music_task(request)
    elif category == MediaCategory.AGENT and isinstance(request, ChatRequest):
        return create_legacy_chat_task(request, category=MediaCategory.AGENT)
    elif category == MediaCategory.TEXT and isinstance(request, ChatRequest):
        return create_legacy_chat_task(request, category=MediaCategory.TEXT)
    elif category == MediaCategory.OMNI and isinstance(request, OmniRequest):
        return create_legacy_omni_task(request)
    elif category == MediaCategory.DETECTION and isinstance(request, DetectionRequest):
        return create_legacy_detection_task(request)

    logger.warning(
        f"不支持的降级任务: category={category}, request_type={type(request)}"
    )
    return None


__all__ = [
    "get_legacy_client",
    "clear_legacy_clients",
    "create_legacy_task",
    "create_legacy_image_task",
    "create_legacy_video_task",
    "create_legacy_speech_task",
    "create_legacy_music_task",
    "create_legacy_chat_task",
    "create_legacy_omni_task",
    "create_legacy_detection_task",
]
