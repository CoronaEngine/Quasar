"""
智能池注册表

提供统一的接口，自动检测账号池模块是否可用：
- 可用时：委托给 ai_pool 的真实池
- 不可用时：降级到旧客户端单例模式

这是工具层访问池系统的唯一入口。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from .category import MediaCategory
from .requests import MediaRequest
from .responses import (
    MediaResult,
    MultiMediaResult,
    ChatResult,
)
from .legacy_fallback import (
    create_legacy_task,
    clear_legacy_clients,
)
from ...ai_modules.providers.configs.dataclasses import ProviderConfig

logger = logging.getLogger(__name__)

# ============================================================================
# 池可用性检测
# ============================================================================

_pool_available: Optional[bool] = None
_pool_check_lock = threading.Lock()


def _check_pool_available() -> bool:
    """
    检测 ai_pool 模块是否可用

    仅检测一次并缓存结果。
    """
    global _pool_available

    if _pool_available is not None:
        return _pool_available

    with _pool_check_lock:
        if _pool_available is not None:
            return _pool_available

        try:
            import service_pool  # noqa: F401
            _pool_available = True
            logger.info("检测到 ai_pool 模块，使用账号池系统")
        except ImportError:
            _pool_available = False
            logger.info(
                "未检测到 ai_pool 模块，使用旧客户端降级模式"
            )

        return _pool_available


def is_pool_mode() -> bool:
    """检查当前是否为池模式（vs 降级模式）"""
    return _check_pool_available()


def reset_pool_detection() -> None:
    """重置池检测状态（用于测试）"""
    global _pool_available
    with _pool_check_lock:
        _pool_available = None


# ============================================================================
# 池初始化辅助
# ============================================================================

_pool_init_attempted = False
_pool_init_lock = threading.Lock()


def _ensure_pool_initialized() -> bool:
    """
    确保账号池已初始化（仅在池模式下）

    此函数会在首次访问池注册表时自动调用，确保账号池已从配置加载。
    使用锁防止并发初始化。

    返回:
    - 是否成功初始化
    """
    global _pool_init_attempted

    if _pool_init_attempted:
        return True

    with _pool_init_lock:
        if _pool_init_attempted:
            return True

        try:
            from service_pool import (
                initialize_account_pools,
                is_pool_initialized,
            )

            if not is_pool_initialized():
                result = initialize_account_pools()
                if result:
                    logger.info("账号池自动初始化成功")
                else:
                    logger.warning("账号池自动初始化失败，部分工具可能不可用")

            _pool_init_attempted = True
            return True

        except Exception as e:
            logger.error(f"账号池初始化异常: {e}")
            _pool_init_attempted = True
            return False


# ============================================================================
# 代理池注册表
# ============================================================================


class _LegacyPoolStub:
    """
    降级模式下的伪池

    用于在工具层检测 get_pool() 时返回非 None，
    表示该类别可用（通过旧客户端）。
    """

    def __init__(self, category: MediaCategory):
        self.category = category
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """检查旧客户端是否可用"""
        if self._available is None:
            from .legacy_fallback import (
                get_legacy_client,
            )

            self._available = get_legacy_client(self.category) is not None
        return self._available


class PoolRegistry:
    """
    代理池注册表

    自动检测并委托给真实池或降级到旧客户端。
    提供与 ai_pool.PoolRegistry 兼容的接口。
    """

    _instance: Optional["PoolRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PoolRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._legacy_stubs: Dict[MediaCategory, _LegacyPoolStub] = {}
        self._initialized = True

        logger.debug("PoolRegistry 代理初始化完成")

    def _get_real_registry(self):
        """获取真实的池注册表（如果可用）"""
        if not _check_pool_available():
            return None

        try:
            from service_pool import get_pool_registry

            # 确保账号池已初始化
            _ensure_pool_initialized()

            return get_pool_registry()
        except Exception as e:
            logger.error(f"获取真实池注册表失败: {e}")
            return None

    def _convert_category(self, category: MediaCategory):
        """
        转换公共项目的 MediaCategory 到 InnerAgentWorkflow 的 MediaCategory

        因为两个模块各自定义了枚举，需要通过值进行转换。
        """
        if not _check_pool_available():
            return category

        try:
            from service_pool import MediaCategory as InnerCategory

            return InnerCategory(category.value)
        except Exception:
            return category

    def get_pool(self, category: MediaCategory) -> Optional[Any]:
        """
        获取指定类别的池

        池模式：委托给真实注册表
        降级模式：返回 LegacyPoolStub（如果旧客户端可用）
        """
        real_registry = self._get_real_registry()

        if real_registry is not None:
            # 转换枚举类型
            inner_category = self._convert_category(category)
            return real_registry.get_pool(inner_category)

        # 降级模式：使用伪池
        if category not in self._legacy_stubs:
            self._legacy_stubs[category] = _LegacyPoolStub(category)

        stub = self._legacy_stubs[category]
        return stub if stub.is_available() else None

    def create_task(
        self,
        category: MediaCategory,
        request: MediaRequest,
    ) -> Optional[Callable[[], Union[MediaResult, MultiMediaResult, ChatResult]]]:
        """
        创建任务

        池模式：委托给真实注册表
        降级模式：使用旧客户端创建任务
        """
        real_registry = self._get_real_registry()

        if real_registry is not None:
            # 转换枚举类型
            inner_category = self._convert_category(category)
            # 转换请求类型
            inner_request = self._convert_request(request)
            return real_registry.create_task(inner_category, inner_request)

        # 降级模式
        task = create_legacy_task(category, request)
        if task is None:
            logger.warning(
                f"降级模式下无法创建任务: category={category.value}, "
                f"请检查旧客户端配置"
            )
        return task

    def _convert_request(self, request: MediaRequest):
        """
        转换公共项目的请求到 InnerAgentWorkflow 的请求类型

        通过重新构造请求对象实现类型转换。
        """
        if not _check_pool_available():
            return request

        try:
            from service_pool import (
                ImageRequest as InnerImageRequest,
                VideoRequest as InnerVideoRequest,
                SpeechRequest as InnerSpeechRequest,
                MusicRequest as InnerMusicRequest,
                ChatRequest as InnerChatRequest,
                OmniRequest as InnerOmniRequest,
                DetectionRequest as InnerDetectionRequest,
            )
            from .requests import (
                ImageRequest,
                VideoRequest,
                SpeechRequest,
                MusicRequest,
                ChatRequest,
                OmniRequest,
                DetectionRequest,
            )

            if isinstance(request, ImageRequest):
                return InnerImageRequest(
                    session_id=request.session_id,
                    prompt=request.prompt,
                    resolution=request.resolution,
                    image_size=request.image_size,
                    image_urls=request.image_urls,
                )
            elif isinstance(request, VideoRequest):
                return InnerVideoRequest(
                    session_id=request.session_id,
                    prompt=request.prompt,
                    image_url=request.image_url,
                    resolution=request.resolution,
                    prompt_extend=request.prompt_extend,
                )
            elif isinstance(request, SpeechRequest):
                return InnerSpeechRequest(
                    session_id=request.session_id,
                    text=request.text,
                    voice_type=request.voice_type,
                    speed_ratio=request.speed_ratio,
                    loudness_ratio=request.loudness_ratio,
                    encoding=request.encoding,
                    sample_rate=request.sample_rate,
                )
            elif isinstance(request, MusicRequest):
                return InnerMusicRequest(
                    session_id=request.session_id,
                    prompt=request.prompt,
                    style=request.style,
                    duration=request.duration,
                    model=request.model,
                )
            elif isinstance(request, ChatRequest):
                return InnerChatRequest(
                    session_id=request.session_id,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            elif isinstance(request, OmniRequest):
                return InnerOmniRequest(
                    session_id=request.session_id,
                    prompt=request.prompt,
                    image_urls=request.image_urls,
                    video_urls=request.video_urls,
                    audio_urls=request.audio_urls,
                )
            elif isinstance(request, DetectionRequest):
                return InnerDetectionRequest(
                    session_id=request.session_id,
                    image_url=request.image_url,
                    target_description=request.target_description,
                )
            else:
                # 未知请求类型，直接返回
                return request
        except Exception as e:
            logger.warning(f"转换请求类型失败: {e}，使用原始请求")
            return request

    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有池的状态统计"""
        real_registry = self._get_real_registry()

        if real_registry is not None:
            return real_registry.get_all_stats()

        # 降级模式：返回简化统计
        return {
            "mode": "legacy_fallback",
            "available_categories": [
                cat.value
                for cat, stub in self._legacy_stubs.items()
                if stub.is_available()
            ],
        }

    def initialize_from_config(self, config: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        从配置初始化池

        池模式：委托给真实注册表
        降级模式：忽略（旧客户端从 AIConfig 自动加载配置）
        """
        real_registry = self._get_real_registry()

        if real_registry is not None:
            real_registry.initialize_from_config(config)
        else:
            logger.info("降级模式下忽略 initialize_from_config，使用 AIConfig 配置")


# ============================================================================
# 模块级便捷函数
# ============================================================================

_registry: Optional[PoolRegistry] = None
_registry_lock = threading.Lock()


def get_pool_registry() -> PoolRegistry:
    """获取 PoolRegistry 单例"""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = PoolRegistry()
    return _registry


def reset_pool_registry() -> None:
    """重置 PoolRegistry（用于测试）"""
    global _registry, _pool_init_attempted
    with _registry_lock:
        _registry = None
        clear_legacy_clients()
        reset_pool_detection()
    with _pool_init_lock:
        _pool_init_attempted = False


def get_category_pool(category: MediaCategory) -> Optional[Any]:
    """便捷函数：获取类别池"""
    return get_pool_registry().get_pool(category)


def create_media_task(
    category: MediaCategory,
    request: MediaRequest,
) -> Optional[Callable[[], Union[MediaResult, MultiMediaResult, ChatResult]]]:
    """便捷函数：创建媒体生成任务"""
    return get_pool_registry().create_task(category, request)


# ============================================================================
# 初始化相关
# ============================================================================

_initialized = False


def initialize_account_pools(
    config: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    *,
    force: bool = False,
) -> bool:
    """
    初始化账号池

    池模式：委托给真实初始化器
    降级模式：总是返回 True（旧客户端按需初始化）
    """
    global _initialized

    if _initialized and not force:
        return True

    if _check_pool_available():
        try:
            from service_pool import (
                initialize_account_pools as real_init,
            )

            result = real_init(config, force=force)
            _initialized = result
            return result
        except Exception as e:
            logger.error(f"初始化账号池失败: {e}")
            return False

    # 降级模式：标记为已初始化
    _initialized = True
    logger.info("降级模式下账号池初始化跳过，使用按需加载的旧客户端")
    return True


def is_pool_initialized() -> bool:
    """检查池是否已初始化"""
    if _check_pool_available():
        try:
            from service_pool import is_pool_initialized as real_check

            return real_check()
        except Exception:
            pass

    return _initialized


def reset_pool_initialization() -> None:
    """重置初始化状态（用于测试）"""
    global _initialized
    _initialized = False

    if _check_pool_available():
        try:
            from service_pool import (
                reset_pool_initialization as real_reset,
            )

            real_reset()
        except Exception:
            pass


# ============================================================================
# LLM 获取（Agent 专用）
# ============================================================================


def get_chat_model(
    *,
    provider_name: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    request_timeout: float | None = None,
    category: str = "agent",  # 默认使用 agent 类别
):
    """
    获取 LLM 客户端（统一入口）

    此函数是 Agent 和工具获取 LLM 的首选方式，自动检测并选择：
    - 池模式：从 ai_pool 的对应池中获取账号并构建 LLM
    - 降级模式：使用 AIConfig 配置的单例客户端

    参数:
    - provider_name: 指定 provider 名称（可选，默认使用 chat.provider）
    - model_name: 指定模型名称（可选，默认使用 chat.model）
    - temperature: 温度参数（可选，默认使用 chat.temperature）
    - request_timeout: 请求超时（可选，默认使用 chat.request_timeout）
    - category: LLM 类别（"agent" 或 "text"），默认 "agent"

    返回:
    - LangChain BaseChatModel 实例

    异常:
    - RuntimeError: 如果无法获取 LLM 客户端
    """
    from ...ai_config.ai_config import (
        get_ai_config,
        # ProviderConfig,
    )
    from ...ai_modules.text_generate.tools.client_openai import build_openai_chat

    config = get_ai_config()
    chat_cfg = config.chat

    # 应用默认值
    provider_key = provider_name or chat_cfg.provider
    model = model_name or chat_cfg.model
    temp = temperature if temperature is not None else chat_cfg.temperature
    timeout = request_timeout if request_timeout is not None else chat_cfg.request_timeout

    # 池模式：尝试从对应池获取
    if _check_pool_available():
        try:
            from service_pool import (
                get_pool_registry as get_real_registry,
                MediaCategory as InnerCategory,
            )

            real_registry = get_real_registry()

            # 根据类别选择对应的池
            if category == "text":
                pool_category = InnerCategory.TEXT
            else:
                pool_category = InnerCategory.AGENT

            llm_pool = real_registry.get_pool(pool_category)

            if llm_pool is not None:
                # 从池中获取账号配置
                release_handle = llm_pool.acquire()
                if release_handle is not None:
                    account = release_handle.account
                    try:
                        # 根据账号配置构建 LLM
                        provider = ProviderConfig(
                            name=account.id,
                            type="openai-compatible",
                            api_key=account.api_key,
                            base_url=account.base_url,
                        )
                        llm = build_openai_chat(
                            provider,
                            model=account.model or model,
                            temperature=temp,
                            request_timeout=timeout,
                        )
                        # 注意：这里不释放账号，因为 LLM 是长期持有的
                        # 后续可以考虑实现更复杂的释放策略
                        logger.debug(
                            f"从 {category.upper()} 池获取 LLM: account={account.id}, "
                            f"model={account.model or model}"
                        )
                        return llm
                    except Exception as e:
                        # 构建失败，释放账号并标记错误
                        release_handle.release(success=False)
                        logger.warning(f"从池构建 LLM 失败: {e}，降级到配置")
        except Exception as e:
            logger.warning(f"访问 {category.upper()} 池失败: {e}，降级到配置")

    # 降级模式：使用 AIConfig
    if provider_key not in config.providers:
        available = ", ".join(sorted(config.providers.keys()))
        raise RuntimeError(
            f"未在配置中找到名为 '{provider_key}' 的 provider。当前可用: {available}"
        )

    provider = config.providers[provider_key]
    if not provider.api_key:
        raise RuntimeError(f"Provider '{provider_key}' 缺少 API Key")

    llm = build_openai_chat(
        provider,
        model=model,
        temperature=temp,
        request_timeout=timeout,
    )
    logger.debug(f"使用降级模式获取 LLM: provider={provider_key}, model={model}")
    return llm


__all__ = [
    # 注册表
    "PoolRegistry",
    "get_pool_registry",
    "reset_pool_registry",
    "get_category_pool",
    "create_media_task",
    # 初始化
    "initialize_account_pools",
    "is_pool_initialized",
    "reset_pool_initialization",
    # 模式检测
    "is_pool_mode",
    "reset_pool_detection",
    # LLM 获取
    "get_chat_model",
]
