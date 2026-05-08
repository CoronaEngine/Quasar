from __future__ import annotations

import logging
from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ....ai_config.ai_config import AIConfig
from ....ai_tools.context import get_current_session
from ....ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)
from ..configs.prompts import OMNI_PROMPTS
from ....ai_models.base_pool import (
    get_pool_registry,
    MediaCategory,
    OmniRequest,
    MediaResult,
)

logger = logging.getLogger(__name__)


class MediaUnderstandingInput(BaseModel):
    """多模态理解输入参数"""

    prompt: str = Field(
        ...,
        description=OMNI_PROMPTS.fields["prompt"],
    )
    image_urls: List[str] = Field(
        default_factory=list,
        description=OMNI_PROMPTS.fields["image_urls"],
    )
    video_urls: List[str] = Field(
        default_factory=list,
        description=OMNI_PROMPTS.fields["video_urls"],
    )
    audio_urls: List[str] = Field(
        default_factory=list,
        description=OMNI_PROMPTS.fields["audio_urls"],
    )


def load_omni_tools(config: AIConfig) -> List[StructuredTool]:
    """
    加载多模态理解工具

    使用 pool 抽象层按账号池分配请求，支持：
    - 多账号加权轮询
    - 熔断与自动恢复
    - 多供应商适配
    """
    omni_cfg = config.omni
    if not omni_cfg.enable:
        return []

    if not omni_cfg.provider or not omni_cfg.model:
        return []

    # 获取池注册表（自动检测账号池或降级模式）
    pool_registry = get_pool_registry()
    if pool_registry.get_pool(MediaCategory.OMNI) is None:
        # 既无账号池也无可用的旧客户端配置
        logger.warning(
            "多模态理解不可用：未配置账号池且旧客户端配置不完整。"
            "请检查 AIConfig 中的 media.omni 和 providers 配置"
        )
        return []

    def _analyze_media(
        prompt: str,
        image_urls: List[str] | None = None,
        video_urls: List[str] | None = None,
        audio_urls: List[str] | None = None,
    ) -> str:
        """多模态内容理解"""
        data = MediaUnderstandingInput(
            prompt=prompt,
            image_urls=image_urls or [],
            video_urls=video_urls or [],
            audio_urls=audio_urls or [],
        )

        # 验证至少有一种媒体输入
        if not any([data.image_urls, data.video_urls, data.audio_urls]):
            return build_error_result(
                error_message="至少需要提供一种媒体输入（图片、视频或音频）"
            ).to_envelope(interface_type="omni")

        # 获取当前 session_id
        session_id = get_current_session()

        # 构建标准请求
        request = OmniRequest(
            session_id=session_id,
            prompt=data.prompt,
            image_urls=data.image_urls,
            video_urls=data.video_urls,
            audio_urls=data.audio_urls,
        )

        # 通过池创建任务
        pool_task = pool_registry.create_task(MediaCategory.OMNI, request)

        if pool_task is None:
            return build_error_result(
                error_message="无可用的多模态理解账号"
            ).to_envelope(interface_type="omni")

        try:
            # 执行任务获取结果
            result: MediaResult = pool_task()

            # 从 metadata 中提取分析结果
            vlm_description = result.metadata.get("analysis_result", "")

            # 构建返回 parts
            parts = [
                build_part(
                    content_type="text",
                    content_text=vlm_description,
                    parameter={
                        "additional_type": ["media_analysis"],
                    },
                )
            ]

            # 返回成功结果
            return build_success_result(parts=parts).to_envelope(interface_type="omni")

        except Exception as e:
            logger.error(f"多模态理解失败: {e}", exc_info=True)
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="omni"
            )

    tool = StructuredTool(
        name="analyze_media",
        description=OMNI_PROMPTS.tool_description,
        args_schema=MediaUnderstandingInput,
        func=_analyze_media,
    )

    return [tool]


__all__ = ["load_omni_tools"]
