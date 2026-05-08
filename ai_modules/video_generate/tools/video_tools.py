"""
视频生成工具
提供基于 LangChain 的视频生成功能（图生视频）
"""

from __future__ import annotations

import os
import logging
from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ....ai_config.ai_config import AIConfig
# MediaToolConfig
from ....ai_models.utils import resize_image_with_constraints
from ....ai_tools.context import get_current_session
from ....ai_media_resource import (
    get_media_registry,
    get_storage_adapter,
    calculate_expire_time,
)
from ....ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)
from ..configs.prompts import VIDEO_PROMPTS
from ....ai_models.base_pool import (
    get_pool_registry,
    MediaCategory,
    VideoRequest,
    MediaResult,
)

logger = logging.getLogger(__name__)


class VideoGenerationInput(BaseModel):
    """视频生成输入参数

    此类定义了图生视频功能所需的所有参数。
    支持基于单张图片和文本提示词生成动态视频内容。
    """

    prompt: str = Field(
        ...,
        description=VIDEO_PROMPTS.fields["prompt"],
    )
    image_url: str = Field(
        ...,
        description=VIDEO_PROMPTS.fields["image_url"],
    )
    resolution: str = Field(
        default="720P",
        description=VIDEO_PROMPTS.fields["resolution"],
    )
    prompt_extend: bool = Field(
        default=True,
        description=VIDEO_PROMPTS.fields["prompt_extend"],
    )


def load_video_tools(config: AIConfig) -> List[StructuredTool]:
    """
    加载视频生成工具

    使用 pool 抽象层按账号池分配请求，支持：
    - 多账号加权轮询
    - 熔断与自动恢复
    - 多供应商适配
    """
    # 获取池注册表（自动检测账号池或降级模式）
    pool_registry = get_pool_registry()
    if pool_registry.get_pool(MediaCategory.VIDEO) is None:
        # 既无账号池也无可用的旧客户端配置
        logger.warning(
            "视频生成不可用：未配置账号池且旧客户端配置不完整。"
            "请检查 AIConfig 中的 media.video 和 providers 配置"
        )
        return []

    media_registry = get_media_registry()
    storage_adapter = get_storage_adapter()

    def _generate_video(
        prompt: str,
        image_url: str,
        resolution: str = "720P",
        prompt_extend: bool = True,
    ) -> str:
        """图生视频：根据图片和提示词生成视频。"""
        data = VideoGenerationInput(
            prompt=prompt,
            image_url=image_url,
            resolution=resolution,
            prompt_extend=prompt_extend,
        )

        # 验证分辨率参数
        valid_resolutions = {"480P", "720P", "1080P"}
        if data.resolution not in valid_resolutions:
            return build_error_result(
                error_message=f"无效的分辨率: {data.resolution}，支持的值: {', '.join(valid_resolutions)}"
            ).to_envelope(interface_type="video")

        # 验证图片 URL 是否存在
        if not data.image_url or not data.image_url.strip():
            return build_error_result(
                error_message="图生视频必须提供图片 URL"
            ).to_envelope(interface_type="video")

        # 准备图片 URL
        input_image_url = data.image_url

        # 如果是 fileid:// URL，解析为真实 URL
        if input_image_url.startswith("fileid://"):
            file_id = input_image_url[9:]  # 提取 "fileid://" 后的部分
            try:
                # 需要原始 HTTP URL（而非 base64），因为上游视频 API 只支持 URL
                input_image_url = media_registry.resolve(
                    file_id, timeout=150.0, return_original_url=True
                )
                logger.debug(f"解析 file_id {file_id} -> {input_image_url}")
            except Exception as e:
                logger.error(f"解析 file_id {file_id} 失败: {e}")
                return build_error_result(
                    error_message=f"无法解析图片 file_id: {e}"
                ).to_envelope(interface_type="video")

        # 如果是本地文件，尝试压缩以避免上传超时
        if input_image_url.startswith("file://"):
            try:
                local_path = input_image_url[7:]
                if os.path.exists(local_path):
                    # 压缩图片 (限制最大边长 1280，兼顾质量和速度)
                    resized_path = resize_image_with_constraints(
                        local_path, max_size=1280
                    )
                    input_image_url = f"file://{resized_path}"
            except Exception as e:
                logger.warning(f"图片压缩失败: {e}")

        # 获取当前 session_id
        session_id = get_current_session()

        # 构建标准请求
        request = VideoRequest(
            session_id=session_id,
            prompt=data.prompt,
            image_url=input_image_url,
            resolution=data.resolution,
            prompt_extend=data.prompt_extend,
        )

        # 通过池创建任务
        pool_task = pool_registry.create_task(MediaCategory.VIDEO, request)

        if pool_task is None:
            return build_error_result(
                error_message="无可用的视频生成账号"
            ).to_envelope(interface_type="video")

        # 定义后台任务：执行池任务 + 存储
        def task():
            # 执行池任务获取 MediaResult
            result: MediaResult = pool_task()

            # 计算过期时间（视频 1 天）
            url_expire_time = result.url_expire_time or calculate_expire_time("video")

            # 通过存储适配器保存（本地模式下载，云端模式直接返回）
            return storage_adapter.save_from_url(
                cloud_url=result.url,
                session_id=session_id,
                resource_type="video",
                url_expire_time=url_expire_time,
            )

        try:
            # 提交任务到注册表，立即返回 file_id
            file_id = media_registry.submit(
                task_fn=task,
                resource_type="video",
                session_id=session_id,
                content_text=data.prompt,
            )

            # 构建 part，使用 file_id（content_url 延迟解析）
            part = build_part(
                content_type="video",
                content_text=data.prompt,
                file_id=file_id,
                parameter={
                    "resolution": data.resolution,
                    "duration": 5,
                },
            )

            # 返回成功结果
            return build_success_result(
                parts=[part],
            ).to_envelope(interface_type="video")

        except Exception as e:
            logger.error(f"视频生成失败: {e}", exc_info=True)
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="video"
            )

    tool = StructuredTool(
        name="generate_video_from_image",
        description=VIDEO_PROMPTS.tool_description,
        args_schema=VideoGenerationInput,
        func=_generate_video,
    )

    return [tool]


__all__ = ["load_video_tools"]
