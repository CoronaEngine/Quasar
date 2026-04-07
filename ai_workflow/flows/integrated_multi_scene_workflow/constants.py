from __future__ import annotations

from typing import Dict, List

MULTI_SCENE_FUNCTION_ID = 21001
IMAGE_MAX_WORKERS = 5

FALLBACK_ELEMENTS: List[Dict[str, str]] = [
    {
        "item_name": "奶油风科技布模块沙发",
        "image_prompt": (
            "A modern creamy-style modular sofa, upholstered in high-tech fabric, "
            "curved soft edges, minimalist design, strictly isolated on pure white "
            "background, soft studio lighting, product photography, 8k resolution, "
            "octane render, masterpiece"
        ),
        "layout_desc": "【位置与动线】放置于客厅视觉中心，呈L型摆放以明确界定会客区，并留出通往阳台的流畅主走道动线。【搭配建议】温润的奶油色调可与原木色茶几或浅色微水泥地面形成呼应，提升空间的柔和与呼吸感。",
    },
    {
        "item_name": "极简黄铜宣纸落地灯",
        "image_prompt": (
            "A wabi-sabi style floor lamp, made of brass and rice paper, "
            "elegant and artistic silhouette, strictly isolated on pure white "
            "background, warm rim lighting, studio photography, photorealistic, "
            "highly detailed"
        ),
        "layout_desc": "【位置与动线】置于沙发侧后方的角落，作为补充照明的同时打破空间的垂直直线条僵硬感。【搭配建议】黄铜金属件与布艺沙发形成材质上的软硬对比，宣纸透出的暖光能极大增强空间的温馨氛围。",
    },
    {
        "item_name": "大幅抽象肌理装饰画",
        "image_prompt": (
            "A framed abstract wall art with heavy impasto texture, neutral earth tones, "
            "modern wabi-sabi style, strictly isolated on pure white "
            "background, bright studio lighting, gallery quality render, masterpiece"
        ),
        "layout_desc": "【位置与动线】悬挂于主沙发正上方的墙面黄金分割点，作为进入该空间的第一视觉焦点（Focal Point）。【搭配建议】画作的泥土色系（Earth tones）提取了空间内的自然元素，使墙面与地面家具在视觉重心上保持平衡。",
    },
]

ANALYZER_SYSTEM_PROMPT = """\
你是一位顶尖的室内空间布局架构师兼资深 AI 绘画（Midjourney/Stable Diffusion）提示词专家。
请根据用户的设计需求（及提供的视觉分析），精准解构出 3-5 个构成该空间的核心设计单品。请注意合理分配单品类型（例如：1个视觉焦点的主家具 + 1-2个辅助家具 + 1-2个氛围灯具或软装配饰）。

针对每个核心单品，请深入思考并提供以下三项内容：
1. item_name —— 中文名称（必须包含材质或风格前缀，如“胡桃木中古风餐边柜”、“透光亚克力茶几”）。
2. image_prompt —— 英文 AI 绘画 Prompt。必须严格按照以下结构编写，以确保生成极高精度的产品白底图：
   [主体精细描述(材质/颜色/形态)] + [设计流派(如 Mid-Century Modern, Wabi-Sabi)] + [摄影光影(studio lighting, soft rim light)] + [环境限定(strictly isolated on pure white background)] + [渲染质量(8k resolution, octane render, masterpiece)]。
3. layout_desc —— 专业的空间布局与搭配规划（中文，约 50-80 字）。必须包含以下两个维度：
   - 【位置与动线】：说明其在三维空间中的物理坐标（如视觉中心、靠窗、转角）及其对空间居住动线（Circulation）的引导作用。
   - 【搭配建议】：说明该物品如何与空间内的其他元素（如背景墙、地板、其他家具）在色彩对比或材质肌理（如冷暖、软硬）上形成和谐搭配。

请 **严格** 以如下 JSON 数组格式输出（绝对不要输出任何 Markdown 代码块外的多余解释文本）：
[
  {
    "item_name": "物品名称",
    "image_prompt": "Detailed description, material, style, isolated on pure white background, studio lighting, octane render, masterpiece",
    "layout_desc": "【位置与动线】...【搭配建议】..."
  }
]
"""

ANALYZER_MULTIMODAL_SUFFIX = (
    "\n\n【参考图片视觉分析】\n"
    "以下是 VLM 对用户提供的参考图片的深度剖析，请严格结合此空间关系与色彩基调来提取设计元素：\n"
)

VLM_ANALYSIS_PROMPT = (
    "你是一位享有盛誉的室内设计视觉分析专家。请以极其专业的眼光深度剖析这张空间图片，并提取核心设计构成：\n"
    "1. 【核心单品解构】：识别出视觉中最具代表性的 3-5 个独立家具或装饰品。详细说明其具体几何形态、主色调、核心材质（如棉麻、黄铜、原木、微水泥）以及工业设计风格。\n"
    "2. 【空间布局与动线】：描述这些核心单品在空间中的相对位置关系、视线焦点（Focal Point）分布，以及它们是如何塑造空间功能分区和物理动线的。\n"
    "3. 【光影与氛围基调】：总结整体空间的风格流派（如侘寂、北欧、法式复古、极简）以及光影分布特征，提炼出关键的色彩与灯光氛围词。\n"
    "请输出高度结构化、专业且精炼的中文分析报告，直接为后续的“单品分离与重建”提供精准的设计依据。"
)
