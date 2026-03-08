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

from ai_config.ai_config import AIConfig, get_ai_config
from ai_models.base_pool import get_chat_model
from ai_tools.registry import get_tool_registry


logger = logging.getLogger(__name__)

_CACHED_AGENT: Any = None
_AGENT_LOCK = threading.RLock()


def _should_retry(error: Exception) -> bool:
    
    if isinstance(error, TypeError):
        error_msg = str(error).lower()
        if "choices" in error_msg or "null" in error_msg:
            return True


    error_msg = str(error).lower()
    retryable_keywords = [
        "timeout", "rate limit", "429", "503", "502", "connection", "temporary",
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
                logger.warning(
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
        from ai_tools.load_tools import load_tools
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


def run_agent(messages: List[BaseMessage]) -> Dict[str, Any]:
    """运行 agent 入口 (同步)"""
    max_account_switches = 2  
    last_error = None
    
    for switch_attempt in range(max_account_switches + 1):
        try:
            
            agent = create_default_agent(force_reload=(switch_attempt > 0))
            return agent.invoke({"messages": messages})
        
        except Exception as e:
            last_error = e
            if switch_attempt < max_account_switches and _is_connection_error(e):
                logger.warning(f"连接错误重试 ({switch_attempt + 1}): {e}")
                time.sleep(1.0) 
            else:
                raise
            
    if last_error:
        
        raise last_error
    return {"messages": []}


def stream_agent(messages: List[BaseMessage]):
    """流式运行 agent 入口 (Stream)"""
    max_account_switches = 2  
    last_error = None
    
    for switch_attempt in range(max_account_switches + 1):
        try:
            
            agent = create_default_agent(force_reload=(switch_attempt > 0))
            
            for chunk in agent.stream({"messages": messages}, stream_mode="updates"):
                
                yield chunk
                
            return
        
        except Exception as e:
            last_error = e
            if switch_attempt < max_account_switches and _is_connection_error(e):
                time.sleep(1.0)  
            else:
                raise
            
    if last_error:
        raise last_error


def fallback_completion(history: List[BaseMessage]) -> str:
    """备用完成方法"""
    cfg = get_ai_config()
    chat_cfg = cfg.chat
    
    llm = get_chat_model(
        provider_name=chat_cfg.provider, model_name=chat_cfg.model,
        temperature=chat_cfg.temperature, request_timeout=chat_cfg.request_timeout,
    )
    prompt_messages: List[BaseMessage] = [SystemMessage(content=chat_cfg.system_prompt), *history]
    ai_message = llm.invoke(prompt_messages)
    content = ai_message.content or ""
    
    if isinstance(content, list):
        content = "\n".join([b["text"] for b in content if b.get("type") == "text"])
        
    return content


__all__ = ["create_default_agent", "run_agent", "stream_agent", "fallback_completion"]