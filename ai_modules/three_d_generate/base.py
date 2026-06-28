from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import tempfile

import httpx

from ...ai_config.ai_config import get_ai_config
from ...ai_service.entrance import register_entrance

from ...ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    session_context,
    extract_parameter,
    parse_tool_response,
    pick_tool,
)
from ...ai_tools.concurrency import session_concurrency
from ...ai_tools.helpers import request_time_diff
from ...ai_tools.request_parser import (
    extract_prompt_from_llm_content,
    extract_images_from_request,
)

from ...ai_tools.session_tracking import (
    init_session,
    update_session_state,
    set_session_error,
)


logger = logging.getLogger(__name__)

# -----------------------------
# 可选：会话追踪（没有则降级为 no-op）
# -----------------------------
def _normalize_llm_content(request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    兼容不同入口字段，统一成 llm_content 结构（list[dict]，每个 dict 里有 part:list）。
    支持：
      - llm_content（标准）
      - llmContent（camelCase）
      - content（部分入口）
      - message.content（integrated 常见）
      - parts（直接给 part 数组）
    """
    llm = request_data.get("llm_content")
    if isinstance(llm, list) and llm:
        return llm

    llm = request_data.get("llmContent")
    if isinstance(llm, list) and llm:
        return llm

    llm = request_data.get("content")
    if isinstance(llm, list) and llm:
        return llm

    msg = request_data.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, list) and c:
            return c

    parts = request_data.get("parts")
    if isinstance(parts, list) and parts:
        return [{"role": "user", "interface_type": "media", "part": parts}]

    return []


def _parse_3d_inputs(request_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    统一解析 3D 输入，返回 (mode, prompt, image_refs)

    image_refs 可能包含：
      - fileid://...
      - http(s)://...
      - 本地路径
    具体如何转成服务端请求格式交给 3D client 去处理。
    """
    llm_content = _normalize_llm_content(request_data)

    image_refs: List[str] = []
    text_chunks: List[str] = []

    # 1) 从 llm_content.part 抽 text/image（模仿 text/video）
    for item in llm_content or []:
        if not isinstance(item, dict):
            continue
        parts = item.get("part") or []
        if not isinstance(parts, list):
            continue

        for p in parts:
            if not isinstance(p, dict):
                continue
            ctype = p.get("content_type")

            if ctype == "text":
                t = p.get("content_text")
                if isinstance(t, str) and t.strip():
                    text_chunks.append(t.strip())

            elif ctype in ("image", "detection", "file"):
                # 有的入口用 content_url，有的用 content_text（fileid:// 常在二者之一）
                u = p.get("content_url") or p.get("content_text")
                if isinstance(u, str) and u.strip():
                    image_refs.append(u.strip())

    # 2) 顶层兼容字段（你之前测试用）
    images = extract_parameter(request_data, "images")
    if isinstance(images, list) and images:
        image_refs.extend([str(x) for x in images if x])

    image_path = extract_parameter(request_data, "image_path") or request_data.get("imagePath")
    if image_path:
        image_refs.append(str(image_path))

    image_url = extract_parameter(request_data, "image_url") or request_data.get("imageUrl")
    if image_url:
        image_refs.append(str(image_url))

    # 3) 顶层文本字段（兼容 message/prompt）
    prompt = extract_parameter(request_data, "prompt")
    if not prompt and isinstance(request_data.get("message"), str):
        prompt = request_data.get("message")
    if not prompt and isinstance(request_data.get("message"), dict):
        prompt = request_data["message"].get("text")

    llm_prompt = " ".join(text_chunks).strip()
    final_prompt = llm_prompt or (str(prompt).strip() if isinstance(prompt, str) else "")

    # 4) 自动判定 mode：图片优先
    if image_refs:
        return "image_to_3d", None, image_refs

    if final_prompt:
        return "text_to_3d", final_prompt, []

    return None, None, []


def _clean_3d_parts(original_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """最小清洗：去掉 None 字段，保持结构兼容你们 response_adapter/前端。"""
    cleaned: List[Dict[str, Any]] = []
    for part in original_parts or []:
        if not isinstance(part, dict):
            continue
        cleaned.append({k: v for k, v in part.items() if v is not None})
    return cleaned


@register_entrance(handler_name="handle_3d_generate")
def handle_3d_generate(payload: Any) -> str:
    request_time_diff(payload)
    request_data: Dict[str, Any] = ensure_dict(payload)

    session_id = request_data.get("session_id") or "default"
    metadata = request_data.get("metadata", {}) or {}
    cfg = get_ai_config()

    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type="media",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )

        return _handle_3d_generate_inner(request_data, session_id, metadata, cfg)


def _handle_3d_generate_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    try:
        logger.debug("3D 生成请求数据：%s", request_data)
        init_session(session_id=session_id, input_type="3d", parameters=request_data)
        update_session_state(session_id, "running")

        # ✅ 关键修复：统一解析输入（不再只认 images/image_path/prompt）
        mode, prompt, img_refs = _parse_3d_inputs(request_data)

        if not mode:
            logger.error(
                "[3D INPUT MISSING] keys=%s llm_content=%s llmContent=%s content=%s message.content=%s parts=%s",
                list(request_data.keys()),
                request_data.get("llm_content"),
                request_data.get("llmContent"),
                request_data.get("content"),
                (request_data.get("message") or {}).get("content") if isinstance(request_data.get("message"), dict) else None,
                request_data.get("parts"),
            )
            raise ValueError("缺少 3D 生成输入：llm_content(text/image) 或 images/image_path/image_url/prompt")

        # 加载 3D tools。当前只使用混元3D。
        from .tools.model_tools import load_hunyuan3d_tools

        # 优先尝试混元3D，如果配置了的话
        tools = []
        tool_name = None
        try:
            hunyuan_tools = load_hunyuan3d_tools(cfg)
            if hunyuan_tools:
                tools = hunyuan_tools
                tool_name = "hunyuan_generate_3d"
                logger.info("使用混元3D引擎")
        except Exception as e:
            logger.warning("混元3D 未配置或加载失败: %s", e)

        if not tools:
            raise RuntimeError("混元 3D 服务不可用，本次模型生成无法继续")

        selected_tool = pick_tool(tools, [tool_name])

        # tool 参数（只传你们 tool 支持的字段）
        tool_params: Dict[str, Any] = {"mode": mode}

        geometry_file_format = extract_parameter(request_data, "geometry_file_format")
        tier = extract_parameter(request_data, "tier")
        object_id = extract_parameter(request_data, "object_id")
        if geometry_file_format:
            tool_params["geometry_file_format"] = geometry_file_format
        if tier:
            tool_params["tier"] = tier
        if object_id:
            tool_params["object_id"] = str(object_id)

        if mode == "image_to_3d":
            # img_refs 可能是 fileid/http/本地路径，由底层 client 统一处理
            tool_params["images"] = img_refs
        else:
            tool_params["prompt"] = prompt

        logger.debug("3d_tool params=%s (engine=%s)", tool_params, tool_name)

        # 调用 tool
        with session_context(session_id) as sid:
            result_json = selected_tool.func(**tool_params)
            session_id = sid

        # 解析 tool 返回
        tool_env = parse_tool_response(result_json)
        if tool_env.get("error_code", 0) != 0:
            raise RuntimeError(tool_env.get("status_info", "3D tool error"))

        llm_content = tool_env.get("llm_content", [])
        if not llm_content:
            raise RuntimeError("3D 生成未返回有效内容")

        original_parts = (llm_content[0] or {}).get("part", [])
        cleaned_parts = _clean_3d_parts(original_parts)
        if not cleaned_parts:
            raise RuntimeError("3D 生成未返回有效 parts")

        # 输出阶段 resolve（fileid:// -> 可访问 URL），保持与 video/text 一致
        from ...ai_tools.response_adapter import resolve_parts
        cleaned_parts = resolve_parts(cleaned_parts, timeout=150.0)

        update_session_state(session_id, "completed")
        return build_success_response(
            interface_type="media",
            session_id=session_id,
            metadata=metadata,
            parts=cleaned_parts,
        )

    except Exception as exc:
        set_session_error(session_id, str(exc))
        update_session_state(session_id, "failed")
        logger.error("3D 生成异常: %s", exc)

        return build_error_response(
            interface_type="media",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )


__all__ = ["handle_3d_generate"]
