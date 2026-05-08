from ....ai_config.prompts import ToolPromptConfig
# ===========================================================================
# 音乐生成提示词
# ===========================================================================

MUSIC_PROMPTS = ToolPromptConfig(
    tool_description=(
        "根据文本提示词生成背景音乐 (BGM)。支持指定模型版本、风格标签；"
        "可选择同步等待生成完成或立即返回任务ID。"
        "返回 JSON 字符串，包含任务ID、状态、可用的音频URL列表。"
    ),
    fields={
        "prompt": "音乐内容或氛围的描述性文本提示词",
        "style": "可选的风格标签，例如 'lofi', 'ambient', 'fantasy', 'epic orchestral' 等",
        "model": "Suno 模型版本，例如: V5, V4_5PLUS, V4_5, V4, V3_5",
        "duration": "期望的音乐时长（秒）。不同模型可能有上限，超出会被后端裁剪或拒绝。",
        "wait_audio": "是否同步等待任务完成。True 将轮询任务详情并在成功后下载音频。False 立即返回任务ID。",
        "max_wait_seconds": "最大等待时间（秒），从CONFIG读取。",
        "poll_interval": "轮询间隔（秒），从CONFIG读取。",
    },
)
