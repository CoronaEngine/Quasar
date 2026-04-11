"""
AI 专属配置
处理 LLM、图像生成、视频生成等 AI 相关配置

本模块是配置的入口点，实际实现分布在子模块中：
- dataclasses/: 所有配置数据类
- loaders/: 配置加载函数
"""

from __future__ import annotations

import copy
import os
import threading

from typing import Any, Dict, Optional

from ai_config.ai_types import AIConfig

# ---------------------------------------------------------------------------
# 模块级缓存
# ---------------------------------------------------------------------------

_AI_CACHE: Optional[AIConfig] = None
_AI_CONFIG_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 配置加载辅助函数
# ---------------------------------------------------------------------------


def _apply_env_overrides(data: Dict[str, Any]) -> None:
    """应用环境变量覆盖"""
    overrides = {
        ("llm", "chat", "model"): os.getenv("CORONA_LLM_MODEL"),
        ("llm", "chat", "provider"): os.getenv("CORONA_LLM_PROVIDER"),
    }
    for path, value in overrides.items():
        if value is None:
            continue
        section = data
        for part in path[:-1]:
            section = section.setdefault(part, {})
        key = path[-1]
        section[key] = value


def _load_ai_config_data() -> Dict[str, Any]:
    """从 ai_settings 模块加载配置"""
    from ai_service.entrance import ai_entrance

    data = copy.deepcopy(ai_entrance.collector.AI_SETTINGS)
    # print(data)
    # data = copy.deepcopy(AI_SETTINGS)
    _apply_env_overrides(data)
    return data


# ---------------------------------------------------------------------------
# 公共函数
# ---------------------------------------------------------------------------


def _build_ai_config() -> AIConfig:
    """构建 AI 配置"""
    _load_ai_config_data()
    # print(raw)
    # providers = _load_providers(raw.get("providers"))
    from ai_service.entrance import get_ai_entrance

    providers = get_ai_entrance().collector.AIConfig.providers
    if not providers:
        raise RuntimeError("AI 配置中至少需要声明一个 provider")

    return get_ai_entrance().collector.AIConfig


def get_ai_config() -> AIConfig:
    """获取 AI 配置（单例，线程安全）"""
    global _AI_CACHE
    if _AI_CACHE is None:
        with _AI_CONFIG_LOCK:
            if _AI_CACHE is None:
                _AI_CACHE = _build_ai_config()
    # print(_AI_CACHE)
    return _AI_CACHE


def reload_ai_config() -> AIConfig:
    """重新加载 AI 配置（线程安全）"""
    global _AI_CACHE
    with _AI_CONFIG_LOCK:
        _AI_CACHE = _build_ai_config()
        try:
            from ai_models.base_pool.registry import reset_pool_registry

            reset_pool_registry()
        except Exception:
            pass
    return _AI_CACHE


__all__ = [
    # 数据类
    "AIConfig",
    # 公共函数
    "get_ai_config",
    "reload_ai_config",
]
