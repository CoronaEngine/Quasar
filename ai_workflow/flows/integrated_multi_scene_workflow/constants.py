from __future__ import annotations

from typing import Dict, List

MULTI_SCENE_FUNCTION_ID = 21001
IMAGE_MAX_WORKERS = 5

FALLBACK_ELEMENTS: List[Dict[str, str]] = [
    {
        "item_name": "现代沙发",
        "image_prompt": (
            "A modern minimalist sofa, clean design, isolated on pure white "
            "background, studio lighting, octane render, masterpiece"
        ),
        "layout_desc": "放置于客厅中央，搭配浅色地毯与茶几。",
    },
    {
        "item_name": "艺术落地灯",
        "image_prompt": (
            "An artistic floor lamp, contemporary design, isolated on white "
            "background, soft studio lighting, product photography, masterpiece"
        ),
        "layout_desc": "置于沙发侧旁，提供柔和氛围照明。",
    },
    {
        "item_name": "装饰画",
        "image_prompt": (
            "A framed abstract wall art, modern style, isolated on pure white "
            "background, studio lighting, high quality render"
        ),
        "layout_desc": "悬挂于沙发上方墙面，作为空间视觉焦点。",
    },
]

ANALYZER_SYSTEM_PROMPT = """\
你是资深室内设计师兼 AI 助手。请根据用户提供的设计需求，构思 3-5 个核心设计单品/元素，
并为每个单品提供：
1. item_name —— 中文名称（简洁明了）
2. image_prompt —— 英文 AI 绘画 Prompt（包含物品描述、风格、纯白背景、产品摄影、\
高质量渲染等关键词）
3. layout_desc —— 该物品在空间中的布局与搭配建议（中文，1-2 句即可）

请 **严格** 以如下 JSON 数组格式输出（不要输出任何多余文本）：
[
  {
    "item_name": "物品名称",
    "image_prompt": "A modern minimalist sofa, clean lines, isolated on pure white \
background, studio lighting, octane render, masterpiece",
    "layout_desc": "放置于客厅中央，搭配浅色地毯与茶几形成会客区。"
  }
]
"""

ANALYZER_MULTIMODAL_SUFFIX = (
    "\n\n【参考图片视觉分析】\n"
    "以下是 VLM 对用户提供的参考图片的分析结果，请结合此信息提取设计元素：\n"
)

VLM_ANALYSIS_PROMPT = (
    "你是室内设计领域的视觉分析专家。请仔细观察图片，描述其中的：\n"
    "1. 主要家具与装饰物品（名称、材质、颜色、风格）\n"
    "2. 空间布局特点（动线、功能分区）\n"
    "3. 整体设计风格与氛围\n"
    "请用结构化的中文描述，便于后续提取设计元素。"
)
