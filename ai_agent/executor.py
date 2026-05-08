"""
Agent 执行模块
负责 Agent 的创建、运行和备用完成逻辑

并发安全说明：
- Agent 本身是无状态的（状态在 messages 中传递）
- 使用 RLock 保护 Agent 创建过程
- 支持多用户并发调用
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call

from ..ai_config.ai_config import AIConfig, get_ai_config
from ..ai_models.base_pool import get_chat_model, is_pool_mode
from ..ai_tools.registry import get_tool_registry


logger = logging.getLogger(__name__)

_CACHED_AGENT: Any = None
_AGENT_LOCK = threading.RLock()  # 保护 Agent 创建


def reset_cached_agent() -> None:
    """清空缓存的 Agent，供配置变更后强制重建。"""
    global _CACHED_AGENT
    with _AGENT_LOCK:
        _CACHED_AGENT = None


def _should_retry(error: Exception) -> bool:

    if isinstance(error, TypeError):
        error_msg = str(error).lower()
        if "choices" in error_msg or "null" in error_msg:
            return True

    error_msg = str(error).lower()
    retryable_keywords = [
        "timeout",
        "rate limit",
        "429",
        "503",
        "502",
        "connection",
        "temporary",
    ]
    return any(keyword in error_msg for keyword in retryable_keywords)


@wrap_model_call
def _retry_middleware(request, handler):

    max_retries = 2
    initial_delay = 1.0
    backoff_factor = 2.0

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return handler(request)
        except Exception as e:
            last_error = e
            if attempt < max_retries and _should_retry(e):
                delay = initial_delay * (backoff_factor**attempt)
                logger.info(
                    f"模型调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，"
                    f"{delay:.1f}秒后重试..."
                )
                time.sleep(delay)
            else:
                raise

    raise last_error  # type: ignore


def _build_agent(config: AIConfig) -> Any:

    chat_cfg = config.chat

    llm = get_chat_model(
        provider_name=chat_cfg.provider,
        model_name=chat_cfg.model,
        temperature=chat_cfg.temperature,
        request_timeout=chat_cfg.request_timeout,
    )

    registry = get_tool_registry()
    if not registry.list_tools():
        from ..ai_tools.load_tools import load_tools

        load_tools(config)

    tools = registry.list_tools()
    logger.debug(f"Agent 使用 {len(tools)} 个工具: {[t.name for t in tools]}")

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=chat_cfg.system_prompt,
        middleware=[_retry_middleware],
    )


def create_default_agent(force_reload: bool = False) -> Any:

    global _CACHED_AGENT

    if _CACHED_AGENT is None or force_reload:

        with _AGENT_LOCK:
            if _CACHED_AGENT is None or force_reload:
                _CACHED_AGENT = _build_agent(get_ai_config())

    return _CACHED_AGENT


def _is_connection_error(error: Exception) -> bool:

    error_msg = str(error).lower()
    connection_keywords = ["connection", "timeout", "unreachable", "refused"]
    return any(keyword in error_msg for keyword in connection_keywords)


def _get_recovery_plan() -> tuple[int, str]:
    """根据当前模式决定连接错误后的恢复策略。"""
    if is_pool_mode():
        return 2, "切换账号重试"
    return 1, "重建客户端重试"


def run_agent(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    运行 agent，接受标准的 LangChain BaseMessage 列表

    当遇到连接错误时，会强制重建 Agent（切换账号）并重试。

    Args:
        messages: LangChain BaseMessage 列表（HumanMessage, AIMessage 等）

    Returns:
        Agent 执行结果 {"messages": [...]}
    """
    max_recovery_attempts, recovery_action = _get_recovery_plan()
    last_error = None

    for recovery_attempt in range(max_recovery_attempts + 1):
        try:
            # 首次尝试后，后续恢复会强制重建 Agent/客户端
            agent = create_default_agent(force_reload=(recovery_attempt > 0))
            return agent.invoke({"messages": messages})

        except Exception as e:
            last_error = e
            if recovery_attempt < max_recovery_attempts and _is_connection_error(e):
                logger.info(
                    f"连接错误，{recovery_action} ({recovery_attempt + 1}/{max_recovery_attempts}): {e}"
                )
                time.sleep(1.0)  # 短暂等待后切换账号
            else:
                raise

    if last_error:

        raise last_error
    return {"messages": []}


def stream_agent(messages: List[BaseMessage]):
    """
    流式运行 agent，逐步返回 AIMessage + 关联的 ToolMessage 组合

    当遇到连接错误时，会强制重建 Agent（切换账号）并重试。

    Args:
        messages: LangChain BaseMessage 列表

    Yields:
        每个 yield 包含：{"messages": [AIMessage, ToolMessage, ...]}
        - 每次 yield 一个完整的推理步骤（AI思考 + 工具调用结果）
    """
    max_recovery_attempts, recovery_action = _get_recovery_plan()
    last_error = None

    for recovery_attempt in range(max_recovery_attempts + 1):
        try:
            # 首次尝试后，后续恢复会强制重建 Agent/客户端
            agent = create_default_agent(force_reload=(recovery_attempt > 0))

            # 使用 stream_mode="updates" 获取每个节点的更新
            for chunk in agent.stream({"messages": messages}, stream_mode="updates"):

                yield chunk

            return

        except Exception as e:
            last_error = e
            if recovery_attempt < max_recovery_attempts and _is_connection_error(e):
                logger.info(
                    f"连接错误，{recovery_action} ({recovery_attempt + 1}/{max_recovery_attempts}): {e}"
                )
                time.sleep(1.0)  # 短暂等待后切换账号
            else:
                raise

    if last_error:
        raise last_error


def fallback_completion(history: List[BaseMessage]) -> str:
    """备用完成方法"""
    cfg = get_ai_config()
    chat_cfg = cfg.chat

    llm = get_chat_model(
        provider_name=chat_cfg.provider,
        model_name=chat_cfg.model,
        temperature=chat_cfg.temperature,
        request_timeout=chat_cfg.request_timeout,
    )
    prompt_messages: List[BaseMessage] = [
        SystemMessage(content=chat_cfg.system_prompt),
        *history,
    ]
    ai_message = llm.invoke(prompt_messages)
    content = ai_message.content or ""

    if isinstance(content, list):
        content = "\n".join([b["text"] for b in content if b.get("type") == "text"])

    return content


__all__ = [
    "create_default_agent",
    "reset_cached_agent",
    "run_agent",
    "stream_agent",
    "fallback_completion",
]
