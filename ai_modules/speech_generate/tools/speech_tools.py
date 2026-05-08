"""
语音合成工具 - 使用 pool 抽象层按账号池分配请求
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ....ai_config.ai_config import AIConfig
from ....ai_tools.context import get_current_session
from ....ai_media_resource import (
    get_media_registry,
    get_storage_adapter,
)
from ....ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)
from ..configs.prompts import SPEECH_PROMPTS
from ....ai_models.base_pool import (
    get_pool_registry,
    MediaCategory,
    SpeechRequest,
    MediaResult,
)

logger = logging.getLogger(__name__)


class SpeechInput(BaseModel):
    """文本转语音的输入参数"""

    text: str = Field(description=SPEECH_PROMPTS.fields["text"])
    voice_type: str = Field(
        default="zh_female_cancan_mars_bigtts",
        description=SPEECH_PROMPTS.fields["voice_type"],
    )
    speed_ratio: float = Field(
        default=1.0,
        description=SPEECH_PROMPTS.fields["speed_ratio"],
    )
    loudness_ratio: float = Field(
        default=1.0,
        description=SPEECH_PROMPTS.fields["loudness_ratio"],
    )
    encoding: str = Field(default="mp3", description=SPEECH_PROMPTS.fields["encoding"])


def load_speech_tools(config: AIConfig):
    """
    加载语音合成工具

    使用 pool 抽象层按账号池分配请求，支持：
    - 多账号加权轮询
    - 熔断与自动恢复
    - 多供应商适配
    """
    # 获取池注册表（自动检测账号池或降级模式）
    pool_registry = get_pool_registry()
    if pool_registry.get_pool(MediaCategory.SPEECH) is None:
        # 既无账号池也无可用的旧客户端配置
        logger.warning(
            "语音合成不可用：未配置账号池且旧客户端配置不完整。"
            "请检查 AIConfig 中的 tts 配置（appid/token）"
        )
        return []

    media_registry = get_media_registry()
    storage_adapter = get_storage_adapter()

    def _text_to_speech(
        text: str,
        voice_type: str = "zh_female_cancan_mars_bigtts",
        speed_ratio: float = 1.0,
        loudness_ratio: float = 1.0,
        encoding: str = "mp3",
    ) -> str:
        """
        文本转语音

        Args:
            text: 待合成的文本
            voice_type: 音色类型
            speed_ratio: 语速比例
            loudness_ratio: 音量比例
            encoding: 音频格式

        Returns:
            JSON 格式的合成结果
        """
        # 验证输入
        if not text or not text.strip():
            return build_error_result(error_message="文本内容不能为空").to_envelope(
                interface_type="speech"
            )

        if len(text) > 1000:
            return build_error_result(
                error_message="文本长度超过1000字符，请分段合成"
            ).to_envelope(interface_type="speech")

        if not (0.1 <= speed_ratio <= 2.0):
            return build_error_result(
                error_message="语速比例应在 0.1 到 2.0 之间"
            ).to_envelope(interface_type="speech")

        if not (0.5 <= loudness_ratio <= 2.0):
            return build_error_result(
                error_message="音量比例应在 0.5 到 2.0 之间"
            ).to_envelope(interface_type="speech")

        # 获取当前 session_id
        session_id = get_current_session()

        # 获取默认采样率
        try:
            from ....ai_config.ai_config import get_ai_config

            ai_config = get_ai_config()
            sample_rate = ai_config.audio.sample_rate
        except Exception:
            sample_rate = 24000

        # 构建标准请求
        request = SpeechRequest(
            session_id=session_id,
            text=text,
            voice_type=voice_type,
            speed_ratio=speed_ratio,
            loudness_ratio=loudness_ratio,
            encoding=encoding,
            sample_rate=sample_rate,
        )

        # 通过池创建任务
        pool_task = pool_registry.create_task(MediaCategory.SPEECH, request)

        if pool_task is None:
            return build_error_result(error_message="无可用的语音合成账号").to_envelope(
                interface_type="speech"
            )

        # 定义后台任务：执行池任务 + 存储
        def task():
            # 执行池任务获取 MediaResult
            result: MediaResult = pool_task()

            # 获取 URL 过期时间
            url_expire_time = result.url_expire_time

            # 通过存储适配器保存（本地模式下载，云端模式直接返回）
            storage_result = storage_adapter.save_from_url(
                cloud_url=result.url,
                session_id=session_id,
                resource_type="audio",
                url_expire_time=url_expire_time,
            )
            return storage_result

        try:
            # 提交任务到注册表，立即返回 file_id
            file_id = media_registry.submit(
                task_fn=task,
                resource_type="audio",
                session_id=session_id,
            )

            # 构建 part，使用 file_id（content_url 延迟解析）
            part = build_part(
                content_type="audio",
                content_text=text,
                file_id=file_id,
                parameter={
                    "additional_type": [voice_type],
                    "duration": 5,
                },
            )

            # 返回成功结果
            return build_success_result(
                parts=[part],
            ).to_envelope(interface_type="speech")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="speech"
            )

    # 创建结构化工具
    tool = StructuredTool.from_function(
        func=_text_to_speech,
        name="text_to_speech",
        description=SPEECH_PROMPTS.tool_description,
        args_schema=SpeechInput,
    )

    return [tool]


__all__ = ["load_speech_tools"]
