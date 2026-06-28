"""
AI 系统预热模块

在应用启动时预热所有缓存和连接，减少首次请求延迟。
设计目标：快速、并行、无阻塞。

预热内容：
1. 配置加载（AI 配置、App 配置）
2. 存储单例（媒体存储、媒体注册表、会话存储、并发管理器）
3. HTTP 客户端连接池（图像、TTS、音乐）
4. 工具注册表（内置工具 + 外部工具）
5. Agent 缓存（包含 LLM 和工具链）
6. 工作流注册表

配置加载优先级：
- 优先加载私有配置
- 回退到默认预设

使用方式：
    from service.warmup import warmup_all
    executor.submit(warmup_all)  # 后台线程执行
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 宣主可注入的 app_config provider
# ---------------------------------------------------------------------------

_app_config_provider: Optional[Callable[[], Any]] = None


def set_app_config_provider(provider: Optional[Callable[[], Any]]) -> None:
    """注入宣主侧的全局应用配置获取器，传 ``None`` 可清除。"""
    global _app_config_provider
    _app_config_provider = provider


def get_app_config_provider() -> Optional[Callable[[], Any]]:
    """获取当前已注入的 app_config 获取器（可能为 None）。"""
    return _app_config_provider


# ---------------------------------------------------------------------------
# 分层预热函数
# ---------------------------------------------------------------------------


def warmup_configs() -> None:
    """
    预热配置单例（第一层，其他组件依赖配置）

    预热：
    - get_ai_config(): AI 专属配置（LLM、媒体工具等）
    - get_app_config(): 全局应用配置（路径等）
    """
    try:
        from ..ai_config.ai_config import get_ai_config

        get_ai_config()
    except Exception as e:
        logger.debug(f"AI 配置预热跳过: {e}")

    if _app_config_provider is not None:
        try:
            _app_config_provider()
        except Exception as e:
            logger.debug(f"App 配置预热跳过: {e}")
    else:
        logger.debug("App 配置预热跳过: 未注入 app_config_provider")


def warmup_storage() -> None:
    """
    预热存储和注册表单例（第二层）

    预热：
    - get_media_store(): 本地媒体存储管理器
    - get_media_registry(): 媒体资源注册表（含线程池）
    - get_conversation_store(): 会话历史存储
    - get_concurrency_manager(): 并发控制管理器
    """
    try:
        from Backend.local_storage.utils import get_media_store

        get_media_store()
    except Exception as e:
        logger.debug(f"媒体存储预热跳过: {e}")

    try:
        from ..ai_media_resource import (
            get_media_registry,
        )

        get_media_registry()
    except Exception as e:
        logger.debug(f"媒体注册表预热跳过: {e}")

    try:
        from ..ai_agent.conversation_store import (
            get_conversation_store,
        )

        get_conversation_store()
    except Exception as e:
        logger.debug(f"会话存储预热跳过: {e}")

    try:
        from .concurrency import (
            get_concurrency_manager,
        )

        get_concurrency_manager()
    except Exception as e:
        logger.debug(f"并发管理器预热跳过: {e}")


def warmup_account_pools() -> None:
    """
    预热账号池（第二层，可与存储并行）

    预热：
    - initialize_account_pools(): 初始化所有媒体类别的账号池

    注意：这会从配置加载账号并注册到池，后续媒体生成工具依赖此池。
    """
    try:
        from ..ai_models.base_pool import (
            initialize_account_pools,
        )

        success = initialize_account_pools()
        if success:
            logger.debug("账号池预热完成")
        else:
            logger.debug("账号池预热跳过: 未配置账号")
    except Exception as e:
        logger.debug(f"账号池预热跳过: {e}")


def warmup_http_clients() -> None:
    """
    预热 HTTP 客户端连接池（第二层，可与存储并行）

    预热：
    - _get_dmx_http_client(): DMX 图像生成 HTTP 客户端
    - _get_tts_http_client(): TTS HTTP 客户端
    - _get_music_http_client(): 音乐生成 HTTP 客户端
    """
    try:
        from ..ai_modules.image_generate.tools.client_dmx import (
            _get_dmx_http_client,
        )

        _get_dmx_http_client()
    except Exception as e:
        logger.debug(f"图像客户端预热跳过: {e}")

    try:
        from ..ai_modules.speech_generate.tools.client_speech import (
            _get_tts_http_client,
        )

        _get_tts_http_client()
    except Exception as e:
        logger.debug(f"TTS 客户端预热跳过: {e}")

    try:
        from ..ai_modules.music_generate.tools.client_music import (
            _get_music_http_client,
        )

        _get_music_http_client()
    except Exception as e:
        logger.debug(f"音乐客户端预热跳过: {e}")


def warmup_tools() -> None:
    """
    预热工具注册表（第二层，可与存储并行）

    预热：
    - load_tools(): 注册内置工具加载器并执行发现
    - 包括内置工具（omni/text/scene）和外部工具（InnerAgentWorkflow/ai_tools）

    注意：这会触发所有工具加载器执行。
    """
    try:
        from . import load_tools
        from ..ai_config.ai_config import get_ai_config

        config = get_ai_config()
        tools = load_tools(config)
        logger.debug(f"工具预热完成: 注册 {len(tools)} 个工具")
    except Exception as e:
        logger.debug(f"工具预热跳过: {e}")


def warmup_agent() -> None:
    """
    预热 Agent 缓存（第三层，依赖配置）

    预热：
    - create_default_agent(): 默认 Agent（包含 LLM 实例和工具链）

    注意：这会创建 LLM 实例，但不会发送实际请求。
    """
    try:
        from ..ai_agent.executor import create_default_agent

        create_default_agent()
    except Exception as e:
        logger.debug(f"Agent 预热跳过: {e}")


def warmup_workflows() -> None:
    """
    预热工作流注册表（第二层，可与存储并行）

    预热：
    - WorkflowRegistry.discover(): 扫描并注册所有工作流

    注意：这会预编译所有 LangGraph StateGraph，便于首次请求快速响应。
    """
    try:
        from ..ai_workflow.registry import (
            get_workflow_registry,
        )

        registry = get_workflow_registry()
        count = registry.discover()
        logger.debug(f"工作流预热完成: 注册 {count} 个工作流")
    except Exception as e:
        logger.debug(f"工作流预热跳过: {e}")


# ---------------------------------------------------------------------------
# 统一预热入口
# ---------------------------------------------------------------------------


def warmup_all(*, parallel: bool = True) -> None:
    """
    预热所有组件（推荐入口）

    参数:
    - parallel: 是否并行预热（默认 True，加速启动）

    预热策略：
    1. 配置必须最先加载（其他组件依赖）
    2. 存储、HTTP 客户端可并行
    3. Agent 最后加载（依赖配置和部分存储）

    总耗时预估：
    - 串行: ~500ms（取决于配置复杂度和网络）
    - 并行: ~200ms
    """
    start_time = time.perf_counter()

    # 第一层：配置（必须先完成）
    warmup_configs()
    config_time = time.perf_counter()

    if parallel:
        # 第二层：存储、HTTP 客户端、工具、工作流、账号池并行
        layer2_tasks: List[Tuple[str, Callable[[], None]]] = [
            ("storage", warmup_storage),
            ("http_clients", warmup_http_clients),
            ("account_pools", warmup_account_pools),
            ("tools", warmup_tools),
            ("workflows", warmup_workflows),
        ]
        with ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="warmup_"
        ) as executor:
            futures = {executor.submit(fn): name for name, fn in layer2_tasks}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"预热 {name} 失败: {e}")
    else:
        # 串行预热
        warmup_storage()
        warmup_http_clients()
        warmup_account_pools()
        warmup_tools()
        warmup_workflows()

    layer2_time = time.perf_counter()

    # 第三层：Agent
    warmup_agent()

    total_time = time.perf_counter() - start_time
    logger.info(
        f"系统预热完成: "
        f"配置 {(config_time - start_time) * 1000:.1f}ms, "
        f"存储/客户端 {(layer2_time - config_time) * 1000:.1f}ms, "
        f"Agent {(time.perf_counter() - layer2_time) * 1000:.1f}ms, "
        f"总计 {total_time * 1000:.1f}ms"
    )


def warmup_minimal() -> None:
    """
    最小预热（仅配置和 Agent）

    适用于需要快速启动的场景，跳过存储和 HTTP 客户端预热。
    """
    start_time = time.perf_counter()

    warmup_configs()
    warmup_agent()

    total_time = time.perf_counter() - start_time
    logger.info(f"最小预热完成: {total_time * 1000:.1f}ms")


__all__ = [
    "warmup_configs",
    "warmup_storage",
    "warmup_http_clients",
    "warmup_account_pools",
    "warmup_tools",
    "warmup_agent",
    "warmup_workflows",
    "warmup_all",
    "warmup_minimal",
]
