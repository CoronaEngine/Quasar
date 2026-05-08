from __future__ import annotations

from langchain_core.tools import tool

from .common import build_success_response
from .context import get_current_session


def _build_text_tool_response(text: str) -> str:
    return build_success_response(
        interface_type="tool",
        session_id=get_current_session(),
        parts=[
            {
                "content_type": "text",
                "content_text": text,
                "parameter": {},
            }
        ],
    )


@tool
def search(query: str) -> str:
    """简易搜索工具，直接回显查询字符串。"""
    return _build_text_tool_response(f"搜索结果（模拟）：{query}")


@tool
def get_weather(location: str) -> str:
    """简易天气工具，返回占位天气。"""
    return _build_text_tool_response(f"{location} 当前天气：晴，22°C（模拟数据）")


def load_test_tools():
    return [search, get_weather]


__all__ = ["load_test_tools"]
