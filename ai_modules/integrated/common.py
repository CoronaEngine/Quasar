from __future__ import annotations

import json
import logging
import time

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage

from ...ai_tools.common import build_success_response
from ...ai_tools.response_adapter import resolve_parts

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """工具执行错误，用于传递工具返回的错误信息。"""

    def __init__(self, message: str, error_code: int = 1):
        super().__init__(message)
        self.error_code = error_code


def is_recoverable_tool_error(error: ToolError) -> bool:
    """判断工具错误是否可降级为提示消息。"""
    error_msg = str(error).lower()
    recoverable_keywords = [
        "connection",
        "timeout",
        "unreachable",
        "refused",
        "temporary",
        "client error",
        "server error",
        "forbidden",
        "not found",
        "bad gateway",
        "rate limit",
        "too many requests",
    ]
    return any(keyword in error_msg for keyword in recoverable_keywords)


def make_assistant_entry(parts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """构造标准 assistant entry。"""
    return {
        "role": "assistant",
        "interface_type": "integrated",
        "sent_time_stamp": int(time.time()),
        "part": parts if parts is not None else [],
    }


def build_tool_error_entry(error: ToolError) -> Dict[str, Any]:
    """将可恢复的工具错误转成普通 assistant 文本消息。"""
    return make_assistant_entry([
        {
            "content_type": "text",
            "content_text": "工具服务暂时不可用，请稍后重试。",
            "content_url": "",
            "parameter": {
                "tool_error": True,
                "recoverable": True,
                "raw_error": str(error),
            },
        }
    ])


def parse_tool_parts(content: str) -> List[Dict[str, Any]]:
    """
    解析工具返回的完整 API 响应 Envelope，提取所有类型的 part。
    如果工具返回了 error_code != 0 的响应，将抛出 ToolError 异常。
    """
    found_parts: List[Dict[str, Any]] = []
    try:
        clean_content = content.strip()
        if clean_content.startswith("```"):
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            else:
                clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]

        data = json.loads(clean_content.strip())
        logger.debug(f"解析工具内容成功: {data}")

        if isinstance(data, dict):
            error_code = data.get("error_code", 0)
            if error_code != 0:
                error_msg = data.get("status_info", "工具执行失败")
                logger.error(f"工具返回错误: error_code={error_code}, status_info={error_msg}")
                raise ToolError(error_msg, error_code)

        if isinstance(data, dict):
            llm_content = data.get("llm_content", [])
            if isinstance(llm_content, list):
                for item in llm_content:
                    parts = item.get("part", [])
                    if isinstance(parts, list):
                        for part in parts:
                            if isinstance(part, dict):
                                content_type = part.get("content_type")
                                if content_type in ["image", "audio", "video", "text", "file", "review"]:
                                    found_parts.append(part)
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f"工具内容解析失败: {exc} 内容: {content}")
    return found_parts


def resolve_tool_message(
    msg: ToolMessage,
    log_prefix: str = "",
) -> Optional[List[Dict[str, Any]]]:
    """
    解析 ToolMessage 并解析 fileid:// URL。

    Returns:
        resolved parts 列表（正常情况）；
        None（可恢复的工具错误，调用方应 yield 错误 entry 后 continue）。
    Raises:
        ToolError: 不可恢复的工具错误。
        RuntimeError: fileid 解析失败。
    """
    try:
        tool_parts = parse_tool_parts(msg.content)
    except ToolError as error:
        if is_recoverable_tool_error(error):
            logger.warning(f"{log_prefix}工具连接失败，降级为提示消息: {error}")
            return None
        logger.error(f"{log_prefix}工具执行失败: {error}")
        raise

    logger.debug(f"{log_prefix}解析 ToolMessage 得到 parts: {tool_parts}")

    try:
        tool_parts = resolve_parts(tool_parts, timeout=150.0)
        logger.debug(f"{log_prefix}解析后的 tool_parts: {tool_parts}")
    except Exception as exc:
        logger.error(f"{log_prefix}解析 tool_parts 中的 file_id 失败: {exc}")
        raise RuntimeError(f"工具资源解析失败: {exc}") from exc

    return tool_parts


def build_success_chunk(
    session_id: str,
    metadata: Dict[str, Any],
    entry: Dict[str, Any],
) -> str:
    """构造单 entry 的成功响应字符串。"""
    return build_success_response(
        interface_type="integrated",
        session_id=session_id,
        metadata=metadata,
        llm_content=[entry],
    )


def build_stream_done_signal(session_id: str, metadata: Dict[str, Any]) -> str:
    """流式结束信号：空 llm_content + metadata.stream_done=True。"""
    return build_success_response(
        interface_type="integrated",
        session_id=session_id,
        metadata={**metadata, "stream_done": True},
        llm_content=[],
    )


def build_heartbeat_signal(session_id: str, metadata: Dict[str, Any]) -> str:
    """心跳信号：空 llm_content + metadata.heartbeat=True。"""
    return build_success_response(
        interface_type="integrated",
        session_id=session_id,
        metadata={**metadata, "heartbeat": True},
        llm_content=[],
    )


def extract_text_parts(msg: AIMessage) -> List[Dict[str, Any]]:
    """从 AIMessage 提取纯文本部分。"""
    content_str = ""

    if isinstance(msg.content, str):
        content_str = msg.content
    elif isinstance(msg.content, list):
        texts = [
            block.get("text", "")
            for block in msg.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content_str = "\n".join(texts)

    if not content_str:
        return []

    parts = [
        {
            "content_type": "text",
            "content_text": content_str,
            "content_url": "",
            "parameter": {},
        }
    ]
    logger.debug(f"提取文本部分: {parts}")
    return parts


__all__ = [
    "ToolError",
    "build_heartbeat_signal",
    "build_stream_done_signal",
    "build_success_chunk",
    "build_tool_error_entry",
    "extract_text_parts",
    "is_recoverable_tool_error",
    "make_assistant_entry",
    "resolve_tool_message",
]
