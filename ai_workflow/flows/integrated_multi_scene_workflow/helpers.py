from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, List

from ai_config.ai_config import get_ai_config
from ai_models.base_pool import MediaCategory, OmniRequest, get_chat_model, get_pool_registry
from ai_tools.registry import get_tool_registry
from ai_tools.response_adapter import FILEID_SCHEME

from .constants import VLM_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


def get_generate_image_tool():
    """惰性加载图片生成工具。"""
    registry = get_tool_registry()
    if not registry.list_tools():
        from ai_tools.load_tools import load_tools

        load_tools(get_ai_config())
    return {t.name: t for t in registry.list_tools()}.get("generate_image")


def extract_text(response: Any) -> str:
    """从 LLM response 中提取纯文本。"""
    content = getattr(response, "content", "")
    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_blocks.append(str(block.get("text", "")))
        return "\n".join(text_blocks)
    return str(content or "")


def extract_image_url(raw_result: Any) -> str:
    """从工具返回值中提取并解析图片 URL（含 fileid 延迟解析）。"""
    try:
        parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        part = parsed["llm_content"][0]["part"][0]
        extracted = str(part.get("content_url") or part.get("content_text") or "")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        extracted = str(raw_result)

    if extracted.count("{") > 1:
        return ""

    if extracted.startswith(FILEID_SCHEME):
        from ai_media_resource import get_media_registry

        file_id = extracted[len(FILEID_SCHEME):]
        try:
            return get_media_registry().resolve(file_id)
        except Exception as e:
            logger.error(
                "[Workflow][generate_images] file_id 解析失败: %s, err=%s",
                file_id,
                e,
            )
            return ""

    return extracted


def to_display_url(url: str) -> str:
    """将本地绝对路径转换为 file:// URL，便于 markdown 展示。"""
    if not url:
        return ""
    lowered = url.lower()
    if lowered.startswith(("http://", "https://", "data:", "file://")):
        return url

    path = Path(url)
    if path.is_absolute():
        return path.as_uri()
    return url


def clean_json_text(raw: str) -> str:
    """去除 markdown code block 包裹，提取纯 JSON 文本。"""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def get_llm(temperature: float = 0.6):
    """获取聊天模型。"""
    cfg = get_ai_config()
    chat_cfg = cfg.chat
    return get_chat_model(
        provider_name=chat_cfg.provider,
        model_name=chat_cfg.model,
        temperature=temperature,
        request_timeout=chat_cfg.request_timeout,
    )


def analyze_images_with_vlm(images: List[str], session_id: str = "") -> str:
    """通过 Omni 模块调用 VLM 对图片进行视觉分析。"""
    valid_images = [u for u in images if (u or "").strip()]
    if not valid_images:
        logger.warning("[Workflow][analyzer] 无可用图片传给 VLM")
        return ""

    try:
        pool_registry = get_pool_registry()
        request = OmniRequest(
            session_id=session_id or f"workflow-{int(time.time())}",
            prompt=VLM_ANALYSIS_PROMPT,
            image_urls=valid_images,
        )
        task = pool_registry.create_task(MediaCategory.OMNI, request)
        if task is None:
            logger.warning("[Workflow][analyzer] Omni 池无可用账号，跳过视觉分析")
            return ""
        result = task()
        analysis = result.metadata.get("analysis_result", "")
        if analysis:
            logger.info("[Workflow][analyzer] VLM 分析完成，结果长度 %s", len(analysis))
        return analysis
    except Exception as e:
        logger.warning("[Workflow][analyzer] VLM 视觉分析失败: %s", e)
        return ""
