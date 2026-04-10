from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ai_config.ai_config import get_ai_config
from ai_tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


def get_tool(name: str) -> Any:
    """从工具注册表中按名称获取工具，按需触发懒加载。"""
    registry = get_tool_registry()
    tools = registry.list_tools()
    if not tools:
        from ai_tools.load_tools import load_tools

        load_tools(get_ai_config())
        tools = registry.list_tools()
    return {t.name: t for t in tools}.get(name)


def parse_tool_result(raw_result: Any) -> Dict[str, Any]:
    """解析工具 envelope，统一返回字典结构。"""
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, str):
        return json.loads(raw_result)
    raise TypeError(f"不支持的工具返回类型: {type(raw_result)!r}")


def extract_tool_error(parsed_result: Dict[str, Any]) -> str:
    """从工具 envelope 中提取错误信息。"""
    error_code = parsed_result.get("error_code", 0)
    if not error_code:
        return ""

    status_info = str(parsed_result.get("status_info", "") or "").strip()
    if status_info and status_info.lower() != "success":
        return status_info

    try:
        parts = parsed_result["llm_content"][0]["part"]
        for part in parts:
            text = str(part.get("content_text", "") or "").strip()
            if text:
                return text
    except (KeyError, IndexError, TypeError):
        pass

    return "工具调用失败"


def parse_placement_result(raw_result: Any) -> Dict[str, Any]:
    """解析 place_scene_from_items 返回值，提取 scene_path 和 actors 列表。"""
    try:
        parsed = parse_tool_result(raw_result)
        error_message = extract_tool_error(parsed)
        if error_message:
            return {"error": error_message}

        parts = parsed.get("llm_content", [{}])[0].get("part", [])
        for part in parts:
            param = part.get("parameter", {})
            scene_path = param.get("scene_path", "")
            actors = param.get("actors", [])
            if scene_path:
                return {"scene_path": scene_path, "actors": actors}
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"error": "scene 布局结果解析失败"}


def parse_import_result(raw_result: Any) -> Dict[str, Any]:
    """解析 import_model 返回值，提取 actor 信息。"""
    try:
        parsed = parse_tool_result(raw_result)
        error_message = extract_tool_error(parsed)
        if error_message:
            return {"error": error_message}

        parts = parsed.get("llm_content", [{}])[0].get("part", [])
        for part in parts:
            text = part.get("content_text", "")
            if text:
                data = json.loads(text)
                if data.get("status") == "success":
                    return data
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"error": "模型导入结果解析失败"}


def parse_review_result(raw_result: Any) -> Dict[str, Any]:
    """解析 scene_rationality_review 返回值。"""
    try:
        parsed = parse_tool_result(raw_result)
        error_message = extract_tool_error(parsed)
        if error_message:
            return {"error": error_message}

        parts = parsed.get("llm_content", [{}])[0].get("part", [])
        for part in parts:
            text = part.get("content_text", "")
            if text:
                data = json.loads(text)
                if "overall" in data:
                    return data
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"error": "场景审查结果解析失败"}
