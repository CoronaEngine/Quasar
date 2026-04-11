from __future__ import annotations

from typing import Any, Dict, List

from ai_workflow.state import MultiSceneWorkflowState
from ai_workflow.streaming import FormatterFunc

from .helpers import to_display_url

NO_OUTPUT: FormatterFunc = lambda _data, _state: []


def format_generate_image_progress_parts(
    *,
    item_name: str,
    image_url: str,
    done_count: int,
    total_count: int,
    error_message: str = "",
) -> List[Dict[str, Any]]:
    """格式化单张图片生成完成时的流式输出。"""
    if error_message:
        text = (
            f"## 图片生成进度\n已完成 **{done_count}/{total_count}** 项\n"
            f"- {item_name}: 生成失败（{error_message}）"
        )
    else:
        text = (
            f"## 图片生成进度\n已完成 **{done_count}/{total_count}** 项\n"
            f"- {item_name}: 图片已生成"
        )

    parts: List[Dict[str, Any]] = [
        {
            "content_type": "text",
            "content_text": text,
            "content_url": "",
            "parameter": {},
        }
    ]

    if image_url:
        parts.append(
            {
                "content_type": "image",
                "content_text": item_name,
                "content_url": to_display_url(image_url),
                "parameter": {},
            }
        )

    return parts


def format_human_review_parts(
    data: Dict[str, Any],
    _state: MultiSceneWorkflowState,
) -> List[Dict[str, Any]]:
    """将审核业务数据格式化为前端 review part。"""
    payload = data.get("review_payload")
    if not isinstance(payload, dict):
        return []

    return [
        {
            "content_type": "review",
            "content_text": "请确认以下设计方案，可编辑后提交。",
            "content_url": "",
            "parameter": {
                "review": payload,
            },
        }
    ]


def format_aggregate_parts(
    data: Dict[str, Any],
    state: MultiSceneWorkflowState,
) -> List[Dict[str, Any]]:
    """将聚合结果格式化为完整的设计方案 parts。"""
    del data
    approved = state.get("approved_elements", [])
    if not approved:
        return []

    generated_images: Dict[str, str] = state.get("generated_images", {})

    parts: List[Dict[str, Any]] = [
        {
            "content_type": "text",
            "content_text": "## 设计方案",
            "content_url": "",
            "parameter": {},
        }
    ]

    for idx, element in enumerate(approved, 1):
        name = element.get("item_name", "未命名")
        desc = element.get("layout_desc", "")

        text_lines = [f"### {idx}. {name}"]
        if desc:
            text_lines.append(desc)
        parts.append(
            {
                "content_type": "text",
                "content_text": "\n".join(text_lines),
                "content_url": "",
                "parameter": {},
            }
        )

        img_url = generated_images.get(name, "")
        if img_url:
            parts.append(
                {
                    "content_type": "image",
                    "content_text": "",
                    "content_url": to_display_url(img_url),
                    "parameter": {},
                }
            )

    return parts
