"""
文本到背景音乐 (BGM) 生成工具

基于 pool 抽象层的音乐生成，支持多账号轮询和熔断恢复。
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ....ai_config.ai_config import AIConfig
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
from ..configs.prompts import MUSIC_PROMPTS
from ....ai_models.base_pool import (
    get_pool_registry,
    MediaCategory,
    MusicRequest,
    MultiMediaResult,
)

logger = logging.getLogger(__name__)


class TextToBGMInput(BaseModel):
    """文本到背景音乐生成输入"""

    prompt: str = Field(..., description=MUSIC_PROMPTS.fields["prompt"])
    style: Optional[str] = Field(
        default=None,
        description=MUSIC_PROMPTS.fields["style"],
    )
    model: str = Field(
        default="V5",
        description=MUSIC_PROMPTS.fields["model"],
    )
    duration: int = Field(
        default=20,
        description=MUSIC_PROMPTS.fields["duration"],
    )


def load_music_tools(config: AIConfig):
    """
    加载音乐生成工具

    使用 pool 抽象层按账号池分配请求，支持：
    - 多账号加权轮询
    - 熔断与自动恢复
    - 多供应商适配
    """
    # 获取池注册表（自动检测账号池或降级模式）
    pool_registry = get_pool_registry()
    if pool_registry.get_pool(MediaCategory.MUSIC) is None:
        # 既无账号池也无可用的旧客户端配置
        logger.warning(
            "音乐生成不可用：未配置账号池且旧客户端配置不完整。"
            "请检查 AIConfig 中的 music 配置（api_key）"
        )
        return []

    media_registry = get_media_registry()
    storage_adapter = get_storage_adapter()

    def _generate_bgm(
        prompt: str,
        style: str | None = None,
        model: str = "V5",
        duration: int = 20,
    ) -> str:
        data = TextToBGMInput(
            prompt=prompt,
            style=style,
            model=model,
            duration=duration,
        )

        if not data.prompt.strip():
            return build_error_result(error_message="提示词不能为空").to_envelope(
                interface_type="music"
            )

        # 获取当前 session_id
        session_id = get_current_session()

        # 构建标准请求
        request = MusicRequest(
            session_id=session_id,
            prompt=data.prompt,
            style=data.style,
            model=data.model,
            duration=data.duration,
        )

        # 通过池创建任务
        pool_task = pool_registry.create_task(MediaCategory.MUSIC, request)

        if pool_task is None:
            return build_error_result(
                error_message="无可用的音乐生成账号"
            ).to_envelope(interface_type="music")

        # 定义后台任务：执行池任务 + 存储（可能返回多个音频）
        def task():
            # 执行池任务获取 MultiMediaResult
            result: MultiMediaResult = pool_task()

            # 计算过期时间（BGM 15 天）
            url_expire_time = calculate_expire_time("audio")

            all_results = result.all_results
            if not all_results:
                raise RuntimeError("未在结果中找到音频数据")

            # 单个音频：直接通过 storage_adapter 保存并返回
            if len(all_results) == 1:
                return storage_adapter.save_from_url(
                    cloud_url=all_results[0].url,
                    session_id=session_id,
                    resource_type="audio",
                    url_expire_time=url_expire_time,
                )

            # 多个音频：第一个作为主音频，其余注册为额外 file_id
            primary = all_results[0]

            for extra_item in all_results[1:]:
                # 为每个额外音频创建任务
                extra_cloud_url = extra_item.url

                def create_task(url: str, exp_time):
                    """创建保存任务的闭包"""

                    def inner_task():
                        return storage_adapter.save_from_url(
                            cloud_url=url,
                            session_id=session_id,
                            resource_type="audio",
                            url_expire_time=exp_time,
                        )

                    return inner_task

                media_registry.submit(
                    task_fn=create_task(extra_cloud_url, url_expire_time),
                    resource_type="audio",
                    session_id=session_id,
                    content_text=extra_item.metadata.get("title", ""),
                )

            # 返回主音频的 StorageResult
            return storage_adapter.save_from_url(
                cloud_url=primary.url,
                session_id=session_id,
                resource_type="audio",
                url_expire_time=url_expire_time,
            )

        try:
            # 提交任务到注册表，立即返回 file_id
            file_id = media_registry.submit(
                task_fn=task,
                resource_type="audio",
                session_id=session_id,
                content_text=data.prompt,
            )

            # 构建 part，使用 file_id（content_url 为 fileid://{id}，延迟解析）
            part = build_part(
                content_type="audio",
                content_text=data.prompt,
                file_id=file_id,
                parameter={
                    "additional_type": [data.style],
                    "duration": data.duration,
                },
            )

            return build_success_result(
                parts=[part],
            ).to_envelope(interface_type="music")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="music"
            )

    tool = StructuredTool(
        name="generate_bgm_music",
        description=MUSIC_PROMPTS.tool_description,
        args_schema=TextToBGMInput,
        func=_generate_bgm,
    )

    return [tool]


__all__ = ["load_music_tools"]
