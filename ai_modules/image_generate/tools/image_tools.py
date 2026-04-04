from __future__ import annotations

from typing import List

from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ai_config.ai_config import AIConfig
# , MediaToolConfig)
from ai_tools.context import get_current_session
from ai_media_resource import (
    get_media_registry,
    get_storage_adapter,
    calculate_expire_time,
)
from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)
from ai_modules.image_generate.configs.prompts import IMAGE_PROMPTS
from ai_models.base_pool import (
    get_pool_registry,
    MediaCategory,
    ImageRequest,
    MediaResult,
)


class ImageGenerationInput(BaseModel):
    """图片生成输入参数

    此类定义了 AI 图片生成功能所需的所有参数。
    支持纯文本生成和基于输入图片列表的合成编辑。
    """

    prompt: str = Field(
        ...,
        description=IMAGE_PROMPTS.fields["prompt"],
    )
    resolution: str = Field(
        default="1:1",
        description=IMAGE_PROMPTS.fields["resolution"],
    )
    image_size: str = Field(
        default="2K",
        description=IMAGE_PROMPTS.fields["image_size"],
    )
    image_urls: List[str] | None = Field(
        default_factory=list,
        description=IMAGE_PROMPTS.fields["image_urls"],
    )


def load_image_tools(config: AIConfig) -> List[StructuredTool]:
    """
    加载图像生成工具

    使用 pool 抽象层按账号池分配请求，支持：
    - 多账号加权轮询
    - 熔断与自动恢复
    - 多供应商适配
    """
    # 获取池注册表（自动检测账号池或降级模式）
    pool_registry = get_pool_registry()
    if pool_registry.get_pool(MediaCategory.IMAGE) is None:
        # 既无账号池也无可用的旧客户端配置
        import logging

        logging.getLogger(__name__).warning(
            "图像生成不可用：未配置账号池且旧客户端配置不完整。"
            "请检查 AIConfig 中的 media.image 和 providers 配置"
        )
        return []

    media_registry = get_media_registry()
    storage_adapter = get_storage_adapter()

    def _generate(
        prompt: str,
        config: RunnableConfig,
        resolution: str = "1:1",
        image_urls: List[str] | None = None,
        image_size: str = "2K",
    ) -> str:
        data = ImageGenerationInput(
            prompt=prompt,
            resolution=resolution,
            image_urls=image_urls or [],
            image_size=image_size,
        )

        # 获取当前 session_id
        session_id = config.get("configurable").get("session_id", None) or get_current_session()

        # 构建标准请求
        request = ImageRequest(
            session_id=session_id,
            prompt=data.prompt,
            resolution=data.resolution,
            image_size=data.image_size,
            image_urls=data.image_urls if data.image_urls else None,
        )

        # 通过池创建任务
        pool_task = pool_registry.create_task(MediaCategory.IMAGE, request)

        if pool_task is None:
            return build_error_result(
                error_message="无可用的图像生成账号"
            ).to_envelope(interface_type="image")

        # 定义后台任务：执行池任务 + 存储
        def task():
            # 执行池任务获取 MediaResult
            result: MediaResult = pool_task()

            # 检查 URL 是否有效
            if not result.url or not result.url.startswith(("http://", "https://")):
                raise RuntimeError(f"图像生成失败: {result.metadata.get('error', '无效的 URL')}")

            # 计算过期时间（图像 2 小时）
            url_expire_time = result.url_expire_time or calculate_expire_time("image")

            # 通过存储适配器保存（本地模式下载，云端模式直接返回）
            return storage_adapter.save_from_url(
                cloud_url=result.url,
                session_id=session_id,
                resource_type="image",
                url_expire_time=url_expire_time,
            )

        try:
            # 提交任务到注册表，立即返回 file_id
            file_id = media_registry.submit(
                task_fn=task,
                resource_type="image",
                session_id=session_id,
                content_text=data.prompt,
            )

            # 构建 part，使用 file_id（content_url 延迟解析）
            part = build_part(
                content_type="image",
                content_text=data.prompt,
                file_id=file_id,
                parameter={
                    "resolution": data.resolution,
                    "image_size": data.image_size,
                },
            )

            # 返回成功结果
            return build_success_result(
                parts=[part],
            ).to_envelope(interface_type="image")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="image"
            )

    tool = StructuredTool(
        name="generate_image",
        description=IMAGE_PROMPTS.tool_description,
        args_schema=ImageGenerationInput,
        func=_generate,
    )
    return [tool]


__all__ = ["load_image_tools"]
