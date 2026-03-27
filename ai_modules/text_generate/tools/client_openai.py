from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from ai_modules.providers.configs.dataclasses import ProviderConfig


# 自定义 User-Agent，避免 OpenAI SDK 默认的 "OpenAI/Python" User-Agent 被某些 API 代理商拦截
# lingya 等代理商会检测 "OpenAI/Python" User-Agent 并返回 402 错误
_CUSTOM_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def build_openai_chat(
    provider: ProviderConfig,
    *,
    model: str,
    temperature: float,
    request_timeout: float,
) -> BaseChatModel:
    if not provider.api_key:
        raise RuntimeError(
            f"Provider '{provider.name}' 缺少 API Key，无法创建 ChatOpenAI 模型。"
        )

    # 合并自定义请求头与 provider 配置的请求头
    headers = {"User-Agent": _CUSTOM_USER_AGENT}
    if provider.headers:
        headers.update(provider.headers)

    return ChatOpenAI(
        model=model,
        api_key=provider.api_key,
        base_url=provider.base_url,
        default_headers=headers,
        temperature=temperature,
        timeout=request_timeout,
        max_retries=0,
    )


__all__ = ["build_openai_chat"]
