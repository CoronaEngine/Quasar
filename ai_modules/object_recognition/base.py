"""
物体识别模块 —— 入口处理器

提供物体入库和物体搜索两个 HTTP 接口入口，
通过 @register_entrance 装饰器注册到 ai_entrance 全局路由。

接口列表:
    - handle_object_store:  物体入库（六面图 → 嵌入向量 → 存储）
    - handle_object_search: 物体搜索（图片/文字 → 嵌入向量 → 检索）
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from ai_config.ai_config import get_ai_config
from ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    parse_tool_response,
)
from ai_tools.concurrency import session_concurrency
from ai_service.entrance import register_entrance
from ai_tools.context import set_current_session, reset_current_session
from ai_tools.helpers import request_time_diff
from ai_tools.request_parser import (
    extract_prompt_from_llm_content,
    extract_images_from_request,
)
from ai_tools.session_tracking import (
    init_session,
    update_session_state,
    set_session_error,
)

logger = logging.getLogger(__name__)

# 模块接口类型标识
INTERFACE_TYPE = "object_recognition"


# ====================================================================== #
#  物体入库接口
# ====================================================================== #


@register_entrance(handler_name="handle_object_store")
def handle_object_store(payload: Any) -> str:
    """
    物体入库接口，将六面图嵌入并存储到向量数据库。

    请求 payload 结构:
    {
        "session_id": "xxx",
        "metadata": {...},
        "object_id": "chair_001",
        "image_paths": ["path1.jpg", "path2.jpg", ...],
        "name": "办公椅",
        "category": "家具",
        "description": "黑色皮质办公转椅"
    }

    或通过 llm_content 传递图片和描述。
    """
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    session_id = request_data.get("session_id") or "default"
    metadata = request_data.get("metadata", {})
    cfg = get_ai_config()

    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type=INTERFACE_TYPE,
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
        return _handle_object_store_inner(request_data, session_id, metadata, cfg)


def _handle_object_store_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    """物体入库内部实现（在并发控制内执行）"""
    token = set_current_session(session_id)

    try:
        init_session(
            session_id=session_id,
            input_type=INTERFACE_TYPE,
            parameters=request_data,
        )
        update_session_state(session_id, "running")
        logger.debug(f"收到物体入库请求: {request_data}")

        # 提取参数
        object_id = request_data.get("object_id")
        if not object_id:
            raise ValueError("缺少必需参数: object_id")

        # 支持从顶层字段或 llm_content 中提取图片路径
        image_paths = request_data.get("image_paths")
        if not image_paths:
            image_paths = extract_images_from_request(request_data)
        if not image_paths:
            raise ValueError("缺少物体图片: 至少需要提供 1 张图片")

        name = request_data.get("name", "")
        category = request_data.get("category", "")

        # 文字描述从 description 字段或 llm_content 中提取
        description = request_data.get("description", "")
        if not description:
            description = extract_prompt_from_llm_content(request_data) or ""

        # 加载物体识别工具并执行入库
        from ai_modules.object_recognition.tools.recognition_tools import (
            load_recognition_tools,
        )

        tools = load_recognition_tools(cfg)
        if not tools:
            raise RuntimeError("物体识别功能未启用或配置不完整")

        # 找到入库工具
        store_tool = None
        for tool in tools:
            if tool.name == "store_object":
                store_tool = tool
                break

        if store_tool is None:
            raise RuntimeError("未找到物体入库工具")

        result_json = store_tool.invoke(
            {
                "object_id": object_id,
                "image_paths": image_paths,
                "name": name,
                "category": category,
                "description": description,
            },
            config={"session_id": session_id},
        )

        logger.debug(f"store_tool 返回: {result_json}")
        tool_envelope = parse_tool_response(result_json)

        if tool_envelope.get("error_code", 0) != 0:
            error_msg = tool_envelope.get("status_info", "未知错误")
            raise RuntimeError(f"物体入库失败: {error_msg}")

        llm_content = tool_envelope.get("llm_content", [])
        if not llm_content:
            raise RuntimeError("物体入库未返回有效内容")

        parts = llm_content[0].get("part", [])

        update_session_state(session_id, "completed")
        return build_success_response(
            interface_type=INTERFACE_TYPE,
            session_id=session_id,
            metadata=metadata,
            parts=parts,
        )

    except Exception as exc:
        logger.error(f"物体入库异常: {exc}")
        set_session_error(session_id, str(exc))
        update_session_state(session_id, "failed")
        return build_error_response(
            interface_type=INTERFACE_TYPE,
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )
    finally:
        reset_current_session(token)


# ====================================================================== #
#  物体搜索接口
# ====================================================================== #


@register_entrance(handler_name="handle_object_search")
def handle_object_search(payload: Any) -> str:
    """
    物体搜索接口，在向量数据库中检索最相似的物体。

    请求 payload 结构:
    {
        "session_id": "xxx",
        "metadata": {...},
        "query_images": ["path1.jpg", ...],
        "query_text": "红色运动鞋",
        "top_k": 5
    }

    支持三种查询模式:
    1. 纯图片: 只提供 query_images
    2. 纯文字: 只提供 query_text
    3. 混合:   同时提供 query_images + query_text
    """
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)
    session_id = request_data.get("session_id") or "default"
    metadata = request_data.get("metadata", {})
    cfg = get_ai_config()

    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type=INTERFACE_TYPE,
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
        return _handle_object_search_inner(request_data, session_id, metadata, cfg)


def _handle_object_search_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    """物体搜索内部实现（在并发控制内执行）"""
    token = set_current_session(session_id)

    try:
        init_session(
            session_id=session_id,
            input_type=INTERFACE_TYPE,
            parameters=request_data,
        )
        update_session_state(session_id, "running")
        logger.debug(f"收到物体搜索请求: {request_data}")

        # 提取查询参数
        query_images = request_data.get("query_images")
        if not query_images:
            query_images = extract_images_from_request(request_data)

        query_text = request_data.get("query_text", "")
        if not query_text:
            query_text = extract_prompt_from_llm_content(request_data) or ""

        if not query_images and not query_text:
            raise ValueError("搜索失败: 至少需要提供查询图片或文字描述")

        top_k = request_data.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1:
            top_k = 5

        # 加载物体识别工具并执行搜索
        from ai_modules.object_recognition.tools.recognition_tools import (
            load_recognition_tools,
        )

        tools = load_recognition_tools(cfg)
        if not tools:
            raise RuntimeError("物体识别功能未启用或配置不完整")

        # 找到搜索工具
        search_tool = None
        for tool in tools:
            if tool.name == "search_similar_object":
                search_tool = tool
                break

        if search_tool is None:
            raise RuntimeError("未找到物体搜索工具")

        result_json = search_tool.invoke(
            {
                "query_images": query_images or [],
                "query_text": query_text,
                "top_k": top_k,
            },
            config={"session_id": session_id},
        )

        logger.debug(f"search_tool 返回: {result_json}")
        tool_envelope = parse_tool_response(result_json)

        if tool_envelope.get("error_code", 0) != 0:
            error_msg = tool_envelope.get("status_info", "未知错误")
            raise RuntimeError(f"物体搜索失败: {error_msg}")

        llm_content = tool_envelope.get("llm_content", [])
        if not llm_content:
            raise RuntimeError("物体搜索未返回有效内容")

        parts = llm_content[0].get("part", [])
        cleaned_parts = _clean_recognition_parts(parts)

        update_session_state(session_id, "completed")
        return build_success_response(
            interface_type=INTERFACE_TYPE,
            session_id=session_id,
            metadata=metadata,
            parts=cleaned_parts,
        )

    except Exception as exc:
        logger.error(f"物体搜索异常: {exc}")
        set_session_error(session_id, str(exc))
        update_session_state(session_id, "failed")
        return build_error_response(
            interface_type=INTERFACE_TYPE,
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )
    finally:
        reset_current_session(token)


# ====================================================================== #
#  结果清洗
# ====================================================================== #


def _clean_recognition_parts(
    original_parts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    清洗物体识别结果 parts。

    保留 content_type、content_text 和 parameter 中的关键字段。
    """
    cleaned_parts = []
    for part in original_parts:
        cleaned_part: Dict[str, Any] = {
            "content_type": part.get("content_type", "text"),
            "content_text": part.get("content_text", ""),
        }

        # 保留 content_url（如果有）
        if "content_url" in part and part["content_url"]:
            cleaned_part["content_url"] = part["content_url"]

        # 清洗 parameter 参数
        original_param = part.get("parameter", {})
        if isinstance(original_param, dict):
            cleaned_param: Dict[str, Any] = {}

            # 保留匹配结果列表
            if "matches" in original_param:
                matches = original_param["matches"]
                if isinstance(matches, list):
                    cleaned_matches = []
                    for match in matches:
                        if isinstance(match, dict):
                            cleaned_match = {}
                            for key in ("rank", "object_id", "name",
                                        "category", "distance", "description"):
                                if key in match:
                                    cleaned_match[key] = match[key]
                            if cleaned_match:
                                cleaned_matches.append(cleaned_match)
                    cleaned_param["matches"] = cleaned_matches

            # 保留统计字段
            for key in ("total", "query_images_count", "query_text",
                        "object_id", "rowid", "image_count", "vector_dim"):
                if key in original_param:
                    cleaned_param[key] = original_param[key]

            if cleaned_param:
                cleaned_part["parameter"] = cleaned_param

        cleaned_parts.append(cleaned_part)

    return cleaned_parts


__all__ = [
    "handle_object_store",
    "handle_object_search",
]
