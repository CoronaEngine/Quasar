"""场景合理性审查工具

在完成多视图拍摄（scene_multi_view_capture）后，将拍摄结果送入多模态大模型
进行场景合理性分析，从四个维度给出结构化评审报告：
  - 物体布局（悬空、穿插、遮挡）
  - 物理合理性（重力支撑、尺寸比例）
  - 风格一致性（材质、光照、整体风格）
  - 整体美观性（构图、密度、空间利用）
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)

logger = logging.getLogger(__name__)

_MAX_IMAGES_DEFAULT = 12  # 默认最大送审图片数，防止 token 超限


# ===========================================================================
# Input Schema
# ===========================================================================

class SceneRationalityReviewInput(BaseModel):
    output_dir: str = Field(
        description=(
            "scene_multi_view_capture 的输出目录路径，"
            "工具将从中读取 PNG 截图送入审查"
        ),
    )
    scene_description: str = Field(
        default="",
        description="场景的预期设计描述，帮助模型理解场景意图（可选）",
    )
    max_images: int = Field(
        default=_MAX_IMAGES_DEFAULT,
        description=(
            f"送入模型的最大图片数量，默认 {_MAX_IMAGES_DEFAULT}。"
            "数量过多会导致 token 超限，建议不超过 20"
        ),
    )


# ===========================================================================
# Helpers
# ===========================================================================

def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _collect_png_files(output_dir: str, max_images: int) -> List[str]:
    """从目录中收集 PNG 文件，按名称排序后均匀间隔采样到 max_images 张。"""
    if not os.path.isdir(output_dir):
        return []
    all_pngs = sorted(
        p for p in os.listdir(output_dir) if p.lower().endswith(".png")
    )
    if not all_pngs:
        return []
    if len(all_pngs) <= max_images:
        return [os.path.join(output_dir, p) for p in all_pngs]
    # 均匀步长采样
    step = len(all_pngs) / max_images
    sampled = [all_pngs[int(i * step)] for i in range(max_images)]
    return [os.path.join(output_dir, p) for p in sampled]


def _parse_json_reply(raw: str) -> dict:
    """尝试从模型回复中提取 JSON，兼容裸 JSON 和代码块包裹两种格式。"""
    raw = raw.strip()
    # 去掉 ```json ... ``` 代码块
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 容错：在回复中搜索第一个完整 JSON 对象
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"raw": raw}


# ===========================================================================
# Tool builder
# ===========================================================================

def _build_scene_rationality_review_tool() -> StructuredTool:

    def _scene_rationality_review(
        *,
        output_dir: str,
        scene_description: str = "",
        max_images: int = _MAX_IMAGES_DEFAULT,
    ) -> str:
        try:
            from openai import OpenAI

            api_key = os.getenv(
                "ZHIZENGZENG_API_KEY",
                "sk-zk249dc3baa72c51a10bb647cbd150953b069d68d09d5f12",
            )
            base_url = "https://api.zhizengzeng.com/v1/"
            client = OpenAI(api_key=api_key, base_url=base_url)

            # --- 收集截图 ---
            png_files = _collect_png_files(output_dir, max_images)
            if not png_files:
                return build_error_result(
                    error_message=f"目录中未找到 PNG 截图：{output_dir}"
                ).to_envelope(interface_type="scene")

            # --- 构建 Prompt ---
            desc_hint = (
                f"场景预期描述：【{scene_description.strip()}】\n"
                if scene_description.strip()
                else ""
            )
            prompt_text = (
                f"{desc_hint}"
                f"以下是一个3D场景的多角度拍摄截图（共 {len(png_files)} 张视角）。\n"
                "请从以下四个维度对场景进行合理性分析：\n"
                "1. **物体布局**：是否存在悬空、穿插、遮挡不合理等问题\n"
                "2. **物理合理性**：重力支撑关系、物体尺寸比例是否符合现实\n"
                "3. **风格一致性**：材质、光照、整体视觉风格是否统一\n"
                "4. **整体美观性**：构图、密度、空间利用是否合理\n\n"
                "请严格以如下 JSON 格式输出分析结果（不要包含任何额外文字）：\n"
                "{\n"
                '  "overall": "PASS" | "NEEDS_IMPROVEMENT" | "FAIL",\n'
                '  "score": <0-100的整数>,\n'
                '  "issues": ["发现的问题1", "发现的问题2"],\n'
                '  "suggestions": ["改进建议1", "改进建议2"],\n'
                '  "details": {\n'
                '    "layout": "布局合理性评价",\n'
                '    "physics": "物理合理性评价",\n'
                '    "style": "风格一致性评价",\n'
                '    "aesthetics": "整体美观性评价"\n'
                "  }\n"
                "}"
            )

            content_list: list = [{"type": "text", "text": prompt_text}]

            # --- 逐张图片编码追加 ---
            loaded_count = 0
            for img_path in png_files:
                try:
                    b64 = _encode_image_to_base64(img_path)
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            },
                        }
                    )
                    loaded_count += 1
                except Exception as e:
                    logger.warning(
                        "[scene_review] 图片读取失败，已跳过 %s: %s", img_path, e
                    )

            if loaded_count == 0:
                return build_error_result(
                    error_message="所有图片读取均失败，无法进行审查"
                ).to_envelope(interface_type="scene")

            logger.info(
                "[scene_review] 共 %d 张图片送入场景合理性审查...", loaded_count
            )

            # --- 调用多模态 VLM ---
            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": content_list}],
                temperature=0.1,
            )
            raw_reply = response.choices[0].message.content.strip()

            review_json = _parse_json_reply(raw_reply)

            result_data = {
                "status": "success",
                "output_dir": output_dir,
                "images_reviewed": loaded_count,
                "review": review_json,
            }
            part = build_part(
                content_type="text",
                content_text=json.dumps(result_data, ensure_ascii=False, indent=2),
            )
            return build_success_result(parts=[part]).to_envelope(
                interface_type="scene"
            )

        except Exception as e:
            logger.error(
                "[scene_review] 场景合理性审查失败: %s", e, exc_info=True
            )
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="scene"
            )

    return StructuredTool(
        name="scene_rationality_review",
        description=(
            "对多视图拍摄结果进行场景合理性分析。"
            "传入 scene_multi_view_capture 的输出目录，工具将均匀采样截图，"
            "通过多模态大模型从布局、物理合理性、风格一致性、整体美观四个维度进行审查，"
            "返回整体评级（PASS / NEEDS_IMPROVEMENT / FAIL）、问题列表和改进建议 JSON。"
            "适合在场景搭建完成后执行质量检验。"
        ),
        args_schema=SceneRationalityReviewInput,
        func=_scene_rationality_review,
    )


# ===========================================================================
# Loader
# ===========================================================================

def load_scene_review_tools() -> List[StructuredTool]:
    return [_build_scene_rationality_review_tool()]


__all__ = ["load_scene_review_tools"]
