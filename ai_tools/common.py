from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional
from collections.abc import Mapping

from ai_agent.conversation import default_session_id
from ai_tools.context import (
    reset_current_session,
    set_current_session,
)


def ensure_dict(payload: Any) -> Dict[str, Any]:
    """确保 payload 为字典。

    支持的输入类型：
    - 已是 dict：直接返回
    - JSON 字符串或 bytes：尝试解析 JSON，若结果为 dict 则返回
    - Mapping（例如 OrderedDict）：转换为 dict 并返回
    其它情况：返回空字典（保持向后兼容）
    """
    # 直接是 dict
    if isinstance(payload, dict):
        return payload

    # 如果是 JSON 字符串
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                return data
            return {}
        except json.JSONDecodeError:
            return {}

    # 如果是 bytes，尝试解码并解析
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = payload.decode()
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            return {}

    # 支持 Mapping 子类（例如 OrderedDict）
    if isinstance(payload, Mapping):
        try:
            return dict(payload)
        except Exception:
            return {}

    # 其它情况保持原有兼容行为：返回空字典
    return {}


def require_fields(data: Dict[str, Any], fields: Iterable[str]) -> None:
    """验证必填字段存在且非空。"""
    missing = [name for name in fields if not data.get(name)]
    if missing:
        raise ValueError(f"缺少必需参数: {', '.join(missing)}")


@contextmanager
def session_context(session_id: Optional[str] = None):
    """统一管理会话上下文，保证 set/reset 成对调用。"""
    sid = session_id or default_session_id()
    token = set_current_session(sid)
    try:
        yield sid
    finally:
        reset_current_session(token)


def has_session_cache(session_id: str) -> bool:
    """检查会话ID是否在AI模块中有缓存。

    Args:
        session_id: 会话ID

    Returns:
        bool: 如果会话存在且有历史记录，返回 True；否则返回 False

    Example:
        >>> if has_session_cache("user-123"):
        >>>     print("会话存在，可以加载历史上下文")
    """
    from ai_agent.conversation_store import (
        get_conversation_store,
    )

    return get_conversation_store().exists(session_id)


def get_session_cache_info(session_id: str) -> Dict[str, Any]:
    """获取会话缓存的详细信息。

    Args:
        session_id: 会话ID

    Returns:
        Dict[str, Any]: 包含以下字段:
            - exists (bool): 会话是否存在
            - message_count (int): 消息数量
            - has_messages (bool): 是否有消息记录

    Example:
        >>> info = get_session_cache_info("user-123")
        >>> if info["has_messages"]:
        >>>     print(f"会话有 {info['message_count']} 条消息")
    """
    from ai_agent.conversation_store import (
        get_conversation_store,
    )

    store = get_conversation_store()

    if not store.exists(session_id):
        return {
            "exists": False,
            "message_count": 0,
            "has_messages": False,
        }

    history = store.snapshot(session_id)
    return {
        "exists": True,
        "message_count": len(history),
        "has_messages": len(history) > 0,
    }


def pick_tool(tools: List[Any], names: Iterable[str]) -> Any:
    """按候选名称顺序选择工具。"""
    for name in names:
        for tool in tools:
            if tool.name == name:
                return tool
    raise RuntimeError(f"未找到匹配的工具: {', '.join(names)}")


def extract_parameter(
    request_data: Dict[str, Any], param_name: str, default: Any = None
) -> Any:
    """从 request_data 或 llm_content 中提取参数。

    优先级：
    1. llm_content 中最后一条 role=user 消息的 part[...]["parameter"] 字段
    2. 兜底：llm_content[0] 的 part[...]["parameter"] 字段
    """

    llm_content = request_data.get("llm_content")
    if isinstance(llm_content, list) and llm_content:
        # 从后往前找最后一条 user 消息（与 extract_user_parts 保持一致）
        for entry in reversed(llm_content):
            if not isinstance(entry, dict):
                continue
            if entry.get("role") == "user":
                parts = entry.get("part", [])
                if isinstance(parts, list):
                    for part in parts:
                        part_params = part.get("parameter", {})
                        if isinstance(part_params, dict) and param_name in part_params:
                            return part_params[param_name]
                break  # 找到了 user 条目但参数不在其中，不继续往前找旧消息

        # 兜底：直接取 llm_content[0]（llm_content 无 role 字段时）
        first = llm_content[0]
        parts = first.get("part", [])
        if isinstance(parts, list):
            for part in parts:
                part_params = part.get("parameter", {})
                if isinstance(part_params, dict) and param_name in part_params:
                    return part_params[param_name]

    return default


def parse_tool_response(response: str | Dict[str, Any]) -> Dict[str, Any]:
    """解析工具返回的响应，支持 JSON 字符串或字典。"""
    if isinstance(response, dict):
        return response
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid tool response format: {str(e)}")


def build_success_response(
    interface_type: str,
    session_id: str,
    metadata: Dict[str, Any] | None = None,
    parts: List[Dict[str, Any]] | None = None,
    role: str = "assistant",
    llm_content: List[Dict[str, Any]] | None = None,
) -> str:
    """构造成功响应结构。

    顶层: session_id, error_code(0), status_info("ok"), llm_content(list), metadata(dict)
    第二层: role, interface_type, sent_time_stamp(int), part(list)
    第三层: part 元素包含 content_type / content_text|content_url / 可选 parameter(dict)

    注意：
    - 当提供 parts 参数时，会自动解析 fileid:// URL（用于独立接口）
    - 当提供 llm_content 参数时，假定已经解析过，不再重复解析（用于流式输出）
    """
    if llm_content is None:
        if parts is None:
            parts = []

        # 自动解析 parts 中的 fileid:// URL（独立接口需要返回真实 URL 给用户）
        from ai_tools.response_adapter import resolve_parts

        try:
            parts = resolve_parts(parts, timeout=150.0)
        except Exception as e:
            # 在最终响应构建阶段，如果解析失败则抛出异常
            # 这会导致上层服务捕获并返回错误响应
            import logging
            logging.getLogger(__name__).error(f"解析 parts 中的 file_id 失败: {e}")
            raise RuntimeError(f"媒体资源解析失败: {e}") from e

        llm_content = [
            {
                "role": role,
                "interface_type": interface_type,
                "sent_time_stamp": int(time.time()),
                "part": parts,
            }
        ]

    body: Dict[str, Any] = {
        "session_id": session_id,
        "error_code": 0,
        "status_info": "ok",
        "llm_content": llm_content,
        "metadata": metadata or {},
    }
    return json.dumps(body, ensure_ascii=False)


def build_error_response(
    interface_type: str,
    session_id: str | None,
    exc: Exception,
    metadata: Dict[str, Any] | None = None,
    role: str = "assistant",
) -> str:
    """构造错误响应结构。

    错误响应也应该符合三层结构：
    - 顶层: error_code=1, status_info=错误信息
    - 第二层: llm_content 包含一个表示错误的消息
    - 第三层: part 包含错误详情的文本
    """
    error_message = str(exc)
    exception_type = type(exc).__name__

    body: Dict[str, Any] = {
        "session_id": session_id or default_session_id(),
        "error_code": 1,
        "status_info": error_message,
        "llm_content": [
            {
                "role": role,
                "interface_type": interface_type,
                "sent_time_stamp": int(time.time()),
                "part": [
                    {
                        "content_type": "text",
                        "content_text": error_message,
                        "parameter": {
                            "error": True,
                            "exception_type": exception_type,
                        },
                    }
                ],
            }
        ],
        "metadata": metadata or {},
    }
    return json.dumps(body, ensure_ascii=False)


__all__ = [
    "ensure_dict",
    "require_fields",
    "session_context",
    "has_session_cache",
    "get_session_cache_info",
    "pick_tool",
    "extract_parameter",
    "parse_tool_response",
    "build_success_response",
    "build_error_response",
]
