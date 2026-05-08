from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ....ai_config.ai_config import AIConfig
from ...providers.configs.dataclasses import ProviderConfig
from .client_openai import build_openai_chat


def get_chat_model(
    config: AIConfig,
    *,
    provider_name: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    request_timeout: float | None = None,
) -> BaseChatModel:
    chat_cfg = config.chat
    provider_key = provider_name or chat_cfg.provider
    provider = _require_provider(config, provider_key)
    model = model_name or chat_cfg.model
    temp = temperature if temperature is not None else chat_cfg.temperature
    timeout = request_timeout if request_timeout is not None else chat_cfg.request_timeout

    if provider.type in {"openai", "openai-compatible"}:
        return build_openai_chat(
            provider,
            model=model,
            temperature=temp,
            request_timeout=timeout,
        )
    raise ValueError(f"暂不支持的模型提供商类型: {provider.type}")


def _require_provider(config: AIConfig, name: str) -> ProviderConfig:
    try:
        return config.providers[name]
    except KeyError as exc:
        available = ", ".join(sorted(config.providers.keys()))
        raise ValueError(
            f"未在 app_config.toml 中找到名为 '{name}' 的 provider。当前可用: {available}"
        ) from exc


__all__ = ["get_chat_model"]
