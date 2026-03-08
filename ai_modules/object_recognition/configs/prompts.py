"""
物体识别模块 —— 工具提示词配置

包含：
- STORE_OBJECT_PROMPTS: 物体入库工具提示词
- SEARCH_OBJECT_PROMPTS: 物体搜索工具提示词
"""

from __future__ import annotations

from ai_config.prompts import ToolPromptConfig


# ── 物体入库工具提示词 ──────────────────────────────────────────────
STORE_OBJECT_PROMPTS = ToolPromptConfig(
    tool_description=(
        "将一个物体存入本地向量数据库，用于后续识别检索。"
        "\n每个物体需要提供 6 张六面图（正面、背面、左侧、右侧、顶部、底部），"
        "系统会将这 6 张图片（可选附带文字描述）融合为单一嵌入向量并持久化存储。"
        "\n支持少于 6 张图片的情况（自动降级处理）。"
    ),
    fields={
        "object_id": (
            "物体唯一标识符，用于区分不同物体。"
            "建议使用有意义的命名，如 'chair_001'、'mug_blue_v2'。"
        ),
        "image_paths": (
            "物体的六面图路径列表（最多 6 张）。"
            "支持本地文件路径、HTTP(S) URL、data: URI。"
            "建议按 [正面, 背面, 左侧, 右侧, 顶部, 底部] 顺序提供。"
        ),
        "name": (
            "物体名称，用于可读性展示。"
            "例如：'蓝色马克杯'、'办公椅A款'。"
        ),
        "category": (
            "物体分类标签，便于分类管理。"
            "例如：'家具'、'餐具'、'电子产品'。"
        ),
        "description": (
            "可选的文字描述，会与图片一起融合到嵌入向量中。"
            "例如：'一把黑色皮质办公转椅，带扶手和滚轮'。"
        ),
    },
)


# ── 物体搜索工具提示词 ──────────────────────────────────────────────
SEARCH_OBJECT_PROMPTS = ToolPromptConfig(
    tool_description=(
        "在本地向量数据库中搜索最相似的物体。"
        "\n支持三种查询方式："
        "\n1. 纯图片查询：提供 1~6 张图片"
        "\n2. 纯文字查询：提供文字描述"
        "\n3. 混合查询：同时提供图片和文字描述"
        "\n返回最相似的物体列表（object_id、名称、距离分数）。"
    ),
    fields={
        "query_images": (
            "查询图片路径列表（0~6 张）。"
            "支持本地文件路径、HTTP(S) URL、data: URI。"
        ),
        "query_text": (
            "查询文字描述。"
            "例如：'红色的运动鞋'、'带金属腿的餐桌'。"
        ),
        "top_k": (
            "返回最相似结果的数量，默认 5。"
        ),
    },
)


__all__ = [
    "STORE_OBJECT_PROMPTS",
    "SEARCH_OBJECT_PROMPTS",
]
