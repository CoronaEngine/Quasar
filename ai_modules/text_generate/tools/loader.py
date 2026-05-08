"""
LLM 配置加载器
"""

from typing import Any, Dict, Mapping

from ..configs.dataclasses import ToolModelConfig, ChatModelConfig
from ....ai_tools.helpers import _as_float
from ....ai_service.entrance import ai_entrance

@ai_entrance.collector.register_loader('chat')
def _load_chat_models(raw: Mapping[str, Any]) -> ChatModelConfig:
    """加载工具模型配置"""

    providers = ai_entrance.collector.AIConfig.providers

    chat = ChatModelConfig(
        provider=str(raw.get("provider", next(iter(providers.keys())))),
        model=str(raw.get("model", "Qwen/Qwen2.5-7B-Instruct")),
        temperature=_as_float(raw.get("temperature", 0.2), 0.2),
        request_timeout=_as_float(raw.get("request_timeout", 60), 60.0),
        system_prompt=str(raw.get("system_prompt", "DEFAULT_SYSTEM_PROMPT")),
    )
    return chat

@ai_entrance.collector.register_loader('tool_models')
def _load_tool_models(raw: Mapping[str, Any]) -> Dict[str, ToolModelConfig]:
    """加载工具模型配置"""
    tool_models: Dict[str, ToolModelConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, Mapping):
            continue
        provider = cfg.get("provider")
        model = cfg.get("model")
        if not provider or not model:
            continue
        tool_models[name] = ToolModelConfig(provider=str(provider), model=str(model))
    return tool_models
