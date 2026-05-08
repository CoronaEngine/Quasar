"""
标准响应模型

定义适配器返回的统一结果格式。
公共项目定义，与 InnerAgentWorkflow 保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MediaResult:
    """
    媒体生成结果

    适配器将供应商 API 响应转换为此标准格式，
    上层通过 StorageAdapter 保存并构建响应 part。
    """

    url: str  # 媒体 URL（云端或本地）
    mime_type: str  # MIME 类型 ("image/png", "video/mp4", ...)
    url_expire_time: Optional[int] = None  # URL 过期时间戳（秒级）
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata 常见字段:
    # - duration: 音视频时长（秒）
    # - resolution: 分辨率 ("720P", "1080P", ...)
    # - title: 标题/名称
    # - actual_prompt: 扩展后的提示词
    # - image_url: 关联封面图 URL


@dataclass
class MultiMediaResult:
    """
    多媒体结果（如音乐生成返回多首）

    primary 为主要结果，extras 为附加结果，
    上层负责将其展开为多个 submit() 调用。
    """

    primary: MediaResult
    extras: List[MediaResult] = field(default_factory=list)

    @property
    def all_results(self) -> List[MediaResult]:
        """获取所有结果（主 + 附加）"""
        return [self.primary] + self.extras


@dataclass
class ChatResult:
    """
    对话/文本生成结果

    LLM 返回的文本内容和 token 统计。
    """

    content: str  # 生成的文本内容
    usage: Optional[Dict[str, int]] = (
        None  # token 统计 {"prompt_tokens": N, "completion_tokens": M, ...}
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 结果转换工具
# ============================================================================


def to_storage_result(result: MediaResult):
    """
    MediaResult -> StorageResult

    对接现有存储适配器。
    """
    from ...ai_media_resource.result import StorageResult

    return StorageResult(url=result.url, url_expire_time=result.url_expire_time)


def build_result_part(
    result: MediaResult,
    content_type: str,
    content_text: str = "",
) -> Dict[str, Any]:
    """
    MediaResult -> part dict

    构建响应 part 结构，对接 response_adapter.build_part。
    """
    from ...ai_tools.response_adapter import build_part

    parameter: Dict[str, Any] = {}

    # 从 metadata 提取常见参数
    if "duration" in result.metadata:
        parameter["duration"] = result.metadata["duration"]
    if "resolution" in result.metadata:
        parameter["resolution"] = result.metadata["resolution"]
    if "additional_type" in result.metadata:
        parameter["additional_type"] = result.metadata["additional_type"]

    return build_part(
        content_type=content_type,
        content_text=content_text,
        content_url=result.url,
        url_expire_time=result.url_expire_time,
        parameter=parameter or None,
    )


__all__ = [
    "MediaResult",
    "MultiMediaResult",
    "ChatResult",
    "to_storage_result",
    "build_result_part",
]
