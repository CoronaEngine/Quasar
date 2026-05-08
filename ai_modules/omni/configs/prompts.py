
from ....ai_config.prompts import ToolPromptConfig


# ==========================================================================
# 多模态理解提示词
# ===========================================================================

OMNI_PROMPTS = ToolPromptConfig(
    tool_description=(
        "多模态内容理解工具，使用 Qwen3-Omni 模型分析图片、视频、音频内容。"
        "\n功能包括："
        "\n- 图片理解：描述图片内容、OCR 文字识别、图片对比分析"
        "\n- 视频理解：视频内容总结、动作识别、场景分析"
        "\n- 音频理解：语音转录、音频内容分析"
        "\n- 多模态融合：同时分析多种媒体类型"
        "\n使用时需提供分析提示词和至少一种媒体 URL。"
        "\n支持内部 fileid:// URL 自动解析。"
    ),
    fields={
        "prompt": (
            "分析提示词，描述你希望 AI 分析什么内容。"
            "例如：'描述这张图片的内容'、'总结这个视频讲了什么'、'转录这段音频的文字'"
        ),
        "image_urls": (
            "可选：图片 URL 列表，支持以下格式："
            "\n- http:// 或 https:// 网络图片 URL"
            "\n- data:image/...;base64,... 格式的 base64 数据 URI"
            "\n- fileid://xxx 内部文件 ID（会自动解析为真实 URL）"
        ),
        "video_urls": (
            "可选：视频 URL 列表，支持以下格式："
            "\n- http:// 或 https:// 网络视频 URL"
            "\n- fileid://xxx 内部文件 ID（会自动解析为真实 URL）"
        ),
        "audio_urls": (
            "可选：音频 URL 列表，支持以下格式："
            "\n- http:// 或 https:// 网络音频 URL"
            "\n- data:audio/...;base64,... 格式的 base64 数据 URI"
            "\n- fileid://xxx 内部文件 ID（会自动解析为真实 URL）"
        ),
    },
)
