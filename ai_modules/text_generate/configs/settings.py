"""
大语言模型配置 - 默认预设

包含：
- DEFAULT_SYSTEM_PROMPT: 默认系统提示词
- LLM_SETTINGS: LLM 模型配置（聊天模型、工具模型等）

实际的模型配置和系统提示词应放在 InnerAgentWorkflow/ai_config/llm.py 中。
"""

from __future__ import annotations

from typing import Any, Dict

# ===========================================================================
# 默认系统提示词（简单版）
# ===========================================================================

DEFAULT_SYSTEM_PROMPT = """你是一个 AI 助手，可以帮助用户完成各种任务。"""

# ===========================================================================
# LLM 模型配置（默认预设）
# ===========================================================================

from ....ai_service.entrance import ai_entrance
# 网络请求配置
@ai_entrance.collector.register_setting("chat")
def CHAT_SETTINGS() ->Dict[str, Any]:
    return {
        "provider": "example",
        "model": "gpt-4",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }

@ai_entrance.collector.register_setting("tool_models")
def TOOL_MODELS_SETTINGS() ->Dict[str, Any]:
    return {
        "mcp": {
            "provider": "example",
            "model": "gpt-4",
        }
    }