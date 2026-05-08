"""
Agent 模块
提供 AI Agent 的核心功能
"""

__all__ = [
    "process_chat_request",
    "create_default_agent",
    "run_agent",
    "fallback_completion",
]


def __getattr__(name):
    if name == "process_chat_request":
        from .interface import process_chat_request

        return process_chat_request
    if name in {"create_default_agent", "run_agent", "fallback_completion"}:
        from .executor import create_default_agent, fallback_completion, run_agent

        return {
            "create_default_agent": create_default_agent,
            "run_agent": run_agent,
            "fallback_completion": fallback_completion,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
