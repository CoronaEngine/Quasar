from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List

from .context import get_current_session

logger = logging.getLogger(__name__)

# 定义上下文变量，用于在工具调用链中隐式传递配置
_interface_type_ctx = ContextVar("interface_type", default=None)
_session_id_ctx = ContextVar("session_id", default=None)


@contextmanager
def tool_context(interface_type: str | None = None, session_id: str | None = None):
    """
    工具执行上下文管理器。
    在调用工具前设置此上下文，工具内部即可自动获取 interface_type 和 session_id。

    Usage:
        with tool_context(interface_type="integrated", session_id="123"):
            result = tool_func(...)
    """
    tokens = {}
    if interface_type is not None:
        tokens[_interface_type_ctx] = _interface_type_ctx.set(interface_type)
    if session_id is not None:
        tokens[_session_id_ctx] = _session_id_ctx.set(session_id)

    try:
        yield
    finally:
        for ctx, token in tokens.items():
            ctx.reset(token)


# fileid:// URL scheme 前缀，用于标识待解析的 file_id
FILEID_SCHEME = "fileid://"


def build_part(
    *,
    content_type: str,
    content_text: str | None = None,
    content_url: str | None = None,
    file_id: str | None = None,
    url_expire_time: int | None = None,
    parameter: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    构建单个 part 结构

    参数:
    - content_type: 内容类型（text/image/video/audio）
    - content_text: 文本内容
    - content_url: 媒体 URL（已解析的真实 URL）
    - file_id: 内部文件 ID（用于延迟解析 URL，会编码为 fileid://{id} 格式存入 content_url）
    - url_expire_time: URL 过期时间（秒级时间戳）
    - parameter: 额外参数

    注意:
    - content_url 和 file_id 互斥，不可同时传入
    - 传入 file_id 时，会自动编码为 fileid://{id} 格式的 content_url
    """
    # 互斥检查：content_url 和 file_id 不可同时传入
    if content_url is not None and file_id is not None:
        raise ValueError(
            "content_url and file_id are mutually exclusive. "
            "Use content_url for resolved URLs, or file_id for deferred resolution."
        )

    part: Dict[str, Any] = {
        "content_type": content_type,
    }
    if content_text is not None:
        part["content_text"] = content_text
    else:
        part["content_text"] = ""

    # file_id 编码为 fileid:// URL scheme
    if file_id is not None:
        part["content_url"] = f"{FILEID_SCHEME}{file_id}"
    elif content_url is not None:
        part["content_url"] = content_url

    if url_expire_time is not None:
        part["url_expire_time"] = url_expire_time
    if parameter:
        filtered_parameter = {k: v for k, v in parameter.items() if v is not None}
        if filtered_parameter:
            part["parameter"] = filtered_parameter
    return part


def resolve_parts(
    parts: List[Dict[str, Any]],
    timeout: float = 150.0,
    encode_to_base64: bool | None = None,
) -> List[Dict[str, Any]]:
    """
    解析 parts 中的 fileid:// URL，填充真实 content_url

    参数:
    - parts: part 列表
    - timeout: 解析超时时间（秒）
    - encode_to_base64: 是否将本地文件 URL 转换为 base64 编码
      - True: 强制转换本地文件为 base64
      - False: 不转换，保持原 URL
      - None: 使用全局配置 (默认)

    返回:
    - 解析后的 parts（多文件场景会展开为多个 part）

    解析逻辑:
    - 识别 content_url 以 fileid:// 开头的 part
    - 调用 media_registry.resolve() 获取真实 URL
    - 如果返回值包含 extra_file_ids，递归解析所有附加文件
    """
    from ..ai_media_resource import get_media_registry

    registry = get_media_registry()
    resolved_parts = []

    for part in parts:
        content_url = part.get("content_url", "")

        # 检查是否为 fileid:// scheme
        if not content_url.startswith(FILEID_SCHEME):
            # 不是 fileid://，直接保留
            resolved_parts.append(part)
            continue

        # 提取 file_id
        file_id = content_url[len(FILEID_SCHEME):]

        try:
            result = registry.resolve_with_expire_time(file_id, encode_to_base64=encode_to_base64)
            resolved_url = result.get("url", "")
            url_expire_time = result.get("url_expire_time")
            logger.debug(f"解析 file_id {file_id} -> url={resolved_url}, url_expire_time={url_expire_time}")

            # 获取完整记录以检查是否有 extra_file_ids
            record = registry.get_by_file_id(file_id)

            # 处理多文件返回格式（通过 parameter 中的 extra_file_ids 判断）
            # 注：这种情况在新架构中较少见，但保留兼容性
            if record and record.parameter and "extra_file_ids" in record.parameter:
                extra_file_ids = record.parameter.get("extra_file_ids", [])
                # 主文件
                part["content_url"] = resolved_url
                if url_expire_time:
                    part["url_expire_time"] = url_expire_time
                if record.content_text:
                    part["content_text"] = record.content_text
                # 更新 parameter 中的元数据
                if "parameter" not in part:
                    part["parameter"] = {}
                for key in ("duration", "image_url"):
                    if key in record.parameter:
                        part["parameter"][key] = record.parameter[key]
                resolved_parts.append(part)

                # 递归解析附加文件
                for extra_info in extra_file_ids:
                    extra_file_id = extra_info.get("file_id")
                    if not extra_file_id:
                        continue

                    # 构建附加 part，使用 fileid:// scheme
                    extra_part = {
                        "content_type": part["content_type"],
                        "content_text": extra_info.get("title", part.get("content_text", "")),
                        "content_url": f"{FILEID_SCHEME}{extra_file_id}",
                    }
                    # 复制基础 parameter 并更新元数据
                    if "parameter" in part:
                        extra_part["parameter"] = {**part["parameter"]}
                    else:
                        extra_part["parameter"] = {}
                    for key in ("duration", "image_url"):
                        if key in extra_info:
                            extra_part["parameter"][key] = extra_info[key]

                    # 递归解析此附加 part
                    resolved_extra = resolve_parts([extra_part], timeout=timeout, encode_to_base64=encode_to_base64)
                    resolved_parts.extend(resolved_extra)

            else:
                # 单个 URL
                part["content_url"] = resolved_url
                if url_expire_time:
                    part["url_expire_time"] = url_expire_time
                resolved_parts.append(part)

        except Exception as e:
            logger.error(f"解析 file_id {file_id} 失败: {e}")
            # 解析失败时抛出异常，让上层服务返回错误响应
            raise RuntimeError(f"解析媒体资源失败 (file_id={file_id}): {e}") from e

    return resolved_parts


class ToolResult:
    """工具内部返回结构"""

    def __init__(
        self,
        *,
        parts: List[Dict[str, Any]],
        metadata: Dict[str, Any] | None = None,
        error_code: int = 0,
        status_info: str = "success",
    ):
        self.parts = parts
        self.metadata = metadata or {}
        self.error_code = error_code
        self.status_info = status_info
        self._resolved = False

    def resolve(self, timeout: float = 150.0, encode_to_base64: bool | None = None) -> "ToolResult":
        """
        解析所有 parts 中的 file_id，填充 content_url

        参数:
        - timeout: 解析超时时间（秒）
        - encode_to_base64: 是否将本地文件 URL 转换为 base64 编码（None 表示使用全局配置）

        返回:
        - self（支持链式调用）
        """
        if not self._resolved:
            self.parts = resolve_parts(self.parts, timeout=timeout, encode_to_base64=encode_to_base64)
            self._resolved = True
        return self

    def to_dict(
        self,
        interface_type: str | None = None,
        session_id: str | None = None,
        role: str = "tools",
        auto_resolve: bool = False,
        encode_to_base64: bool | None = None,
    ) -> Dict[str, Any]:
        """
        转换为字典对象（不进行 JSON 序列化）。
        优先使用上下文中的值，其次使用传入参数。

        参数:
        - interface_type: 接口类型
        - session_id: 会话 ID
        - role: 角色
        - auto_resolve: 是否自动解析 file_id（默认 False，仅在返回用户时设为 True）
        - encode_to_base64: 是否将本地文件 URL 转换为 base64 编码（None 表示使用全局配置）
        """
        # 自动解析 file_id
        if auto_resolve and not self._resolved:
            self.resolve(encode_to_base64=encode_to_base64)

        # 1. 确定 interface_type
        # 优先使用 Context (Agent 强制覆盖)，其次是参数 (工具默认)，最后报错
        final_interface_type = _interface_type_ctx.get() or interface_type
        if final_interface_type is None:
            # 如果既没有传参也没有上下文，为了兼容性暂时允许，但在严格模式下应报错
            # 这里抛出异常以强制规范化
            raise ValueError(
                "interface_type is missing. Please provide it via argument or use 'with tool_context(...):'"
            )

        # 2. 确定 session_id
        sid = _session_id_ctx.get() or session_id or get_current_session()
        # 统一使用秒级时间戳，与其他服务（common.py, base.py）保持一致
        sent_time = int(time.time())

        return {
            "session_id": sid,
            "error_code": self.error_code,
            "status_info": self.status_info,
            "llm_content": [
                {
                    "role": role,
                    "interface_type": final_interface_type,
                    "sent_time_stamp": sent_time,
                    "part": self.parts,
                }
            ],
            "metadata": self.metadata,
        }

    def to_envelope(
        self,
        interface_type: str | None = None,
        session_id: str | None = None,
        role: str = "tools",
        auto_resolve: bool = False,
        encode_to_base64: bool | None = None,
    ) -> str:
        """
        转换为最终 envelope JSON 字符串。
        注意：如果追求性能，建议直接使用 to_dict() 获取对象，避免重复序列化。

        参数:
        - interface_type: 接口类型
        - session_id: 会话 ID
        - role: 角色
        - auto_resolve: 是否自动解析 file_id（默认 False，仅在返回用户时设为 True）
        - encode_to_base64: 是否将本地文件 URL 转换为 base64 编码（None 表示使用全局配置）
        """
        data = self.to_dict(
            interface_type=interface_type,
            session_id=session_id,
            role=role,
            auto_resolve=auto_resolve,
            encode_to_base64=encode_to_base64,
        )
        return json.dumps(data, ensure_ascii=False)


def build_success_result(
    *,
    parts: List[Dict[str, Any]],
    metadata: Dict[str, Any] | None = None,
) -> ToolResult:
    """构建成功的工具结果"""
    return ToolResult(parts=parts, metadata=metadata, error_code=0, status_info="success")


def build_error_result(
    *,
    error_message: str,
    error_code: int = 1,
    metadata: Dict[str, Any] | None = None,
) -> ToolResult:
    """构建错误的工具结果"""
    return ToolResult(
        parts=[build_part(content_type="text", content_text=error_message)],
        metadata=metadata,
        error_code=error_code,
        status_info=error_message,
    )


__all__ = [
    "FILEID_SCHEME",
    "tool_context",
    "build_part",
    "resolve_parts",
    "ToolResult",
    "build_success_result",
    "build_error_result",
]
