"""
工具注册表

管理 tool_name → StructuredTool 的映射，支持：
1. 手动注册单个工具
2. 按分类组织工具（media/text/scene/external）
3. 声明式依赖（工具可声明所需的服务/配置）
4. 自动发现外部工具（InnerAgentWorkflow/ai_tools）

分类说明：
- media: 媒体生成工具（图像、视频、语音、音乐、检测等）
- text: 文案生成工具（产品文案、营销文案、创意文案）
- scene: 场景操作工具（查询、变换等）
- external: 外部/私有工具（来自 InnerAgentWorkflow 等）

使用示例:
    # 获取注册表
    registry = get_tool_registry()

    # 注册工具
    registry.register(my_tool, category="media")

    # 获取单个工具
    tool = registry.get("detect_objects")

    # 按分类获取工具
    media_tools = registry.get_by_category("media")

    # 获取所有工具
    all_tools = registry.list_tools()
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from ..ai_config.ai_config import AIConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# 工具分类
# ===========================================================================


class ToolCategory(str, Enum):
    """工具分类枚举"""

    MEDIA = "media"  # 媒体生成（图像、视频、语音、音乐、检测）
    TEXT = "text"  # 文案生成
    SCENE = "scene"  # 场景操作
    EXTERNAL = "external"  # 外部/私有工具
    TEST = "test"  # 测试工具
    OTHER = "other"  # 其他


# ===========================================================================
# 依赖声明
# ===========================================================================


class DependencyType(str, Enum):
    """依赖类型枚举"""

    # 服务依赖
    MEDIA_REGISTRY = "media_registry"  # 媒体注册表服务
    STORAGE_ADAPTER = "storage_adapter"  # 存储适配器
    SCENE_SERVICE = "scene_service"  # 场景服务

    # 配置依赖
    CONFIG_PROVIDER = "config_provider"  # 需要特定的 API provider
    CONFIG_MEDIA = "config_media"  # 需要媒体配置
    CONFIG_TTS = "config_tts"  # 需要 TTS 配置
    CONFIG_MUSIC = "config_music"  # 需要音乐配置

    # HTTP 客户端依赖
    HTTP_IMAGE = "http_image"  # 图像生成 HTTP 客户端
    HTTP_TTS = "http_tts"  # TTS HTTP 客户端
    HTTP_MUSIC = "http_music"  # 音乐生成 HTTP 客户端
    HTTP_VIDEO = "http_video"  # 视频生成 HTTP 客户端

    # 其他依赖
    LLM = "llm"  # 需要 LLM 模型
    VLM = "vlm"  # 需要视觉语言模型


@dataclass(frozen=False)
class ToolDependency:
    """工具依赖声明

    声明工具运行所需的依赖项。依赖可以是：
    - 服务（如 media_registry）
    - 配置（如特定的 provider）
    - HTTP 客户端（如 http_image）

    Attributes:
        type: 依赖类型
        required: 是否必需（False 表示可选依赖）
        provider: 当 type 为 CONFIG_PROVIDER 时，指定需要的 provider 名称
        config_path: 配置路径（如 "media.detection"）
    """

    type: DependencyType
    required: bool = True
    provider: Optional[str] = None
    config_path: Optional[str] = None

    def __str__(self) -> str:
        parts = [self.type.value]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.config_path:
            parts.append(f"path={self.config_path}")
        if not self.required:
            parts.append("optional")
        return f"Dependency({', '.join(parts)})"


@dataclass
class ToolMetadata:
    """工具元数据

    包含工具的分类、依赖声明等信息。

    Attributes:
        name: 工具名称
        category: 工具分类
        dependencies: 依赖声明列表
        source: 工具来源（模块路径）
        description: 工具描述（可选，默认从 tool.description 获取）
        tags: 自定义标签（用于更细粒度的分类）
    """

    name: str
    category: ToolCategory
    dependencies: List[ToolDependency] = field(default_factory=list)
    source: str = ""
    description: str = ""
    tags: Set[str] = field(default_factory=set)

    def has_dependency(self, dep_type: DependencyType) -> bool:
        """检查是否有指定类型的依赖"""
        return any(d.type == dep_type for d in self.dependencies)

    def get_required_providers(self) -> List[str]:
        """获取所有必需的 provider 名称"""
        providers = []
        for dep in self.dependencies:
            if dep.type == DependencyType.CONFIG_PROVIDER and dep.provider:
                providers.append(dep.provider)
        return providers


# ===========================================================================
# 工具加载器协议
# ===========================================================================

# 工具加载函数类型：接收 AIConfig，返回工具列表
ToolLoaderFunc = Callable[["AIConfig"], List["BaseTool"]]

# 无配置工具加载函数类型
SimpleToolLoaderFunc = Callable[[], List["BaseTool"]]


@dataclass
class ToolLoaderSpec:
    """工具加载器规格

    描述如何加载一组工具。

    Attributes:
        loader: 加载函数（接收 AIConfig 或无参数）
        category: 工具分类
        dependencies: 共享依赖声明（应用于所有加载的工具）
        requires_config: 是否需要 AIConfig 参数
        source: 加载器来源（模块路径）
    """

    loader: Union[ToolLoaderFunc, SimpleToolLoaderFunc]
    category: ToolCategory
    dependencies: List[ToolDependency] = field(default_factory=list)
    requires_config: bool = True
    source: str = ""


# ===========================================================================
# 工具注册表
# ===========================================================================

# 单例缓存
_REGISTRY_INSTANCE: Optional["ToolRegistry"] = None
_REGISTRY_LOCK = threading.Lock()


class ToolRegistry:
    """工具注册表

    线程安全的单例类，管理 tool_name → (StructuredTool, ToolMetadata) 映射。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, "BaseTool"] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._loaders: List[ToolLoaderSpec] = []
        self._lock = threading.RLock()
        self._discovered = False

    # -----------------------------------------------------------------------
    # 注册方法
    # -----------------------------------------------------------------------

    def register(
        self,
        tool: "BaseTool",
        *,
        category: Union[ToolCategory, str] = ToolCategory.OTHER,
        dependencies: Optional[List[ToolDependency]] = None,
        source: str = "",
        tags: Optional[Set[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """注册单个工具

        Args:
            tool: LangChain BaseTool/StructuredTool 实例
            category: 工具分类
            dependencies: 依赖声明列表
            source: 工具来源模块
            tags: 自定义标签
            overwrite: 是否覆盖已存在的注册

        Raises:
            ValueError: 当工具名已存在且 overwrite=False
        """
        if isinstance(category, str):
            category = ToolCategory(category)

        with self._lock:
            if tool.name in self._tools and not overwrite:
                raise ValueError(
                    f"Tool '{tool.name}' already registered. "
                    f"Use overwrite=True to replace."
                )

            self._tools[tool.name] = tool
            self._metadata[tool.name] = ToolMetadata(
                name=tool.name,
                category=category,
                dependencies=dependencies or [],
                source=source,
                description=getattr(tool, "description", ""),
                tags=tags or set(),
            )
            logger.debug(f"Registered tool: {tool.name} (category={category.value})")

    def register_loader(
        self,
        loader: Union[ToolLoaderFunc, SimpleToolLoaderFunc],
        *,
        category: Union[ToolCategory, str] = ToolCategory.OTHER,
        dependencies: Optional[List[ToolDependency]] = None,
        requires_config: bool = True,
        source: str = "",
    ) -> None:
        """注册工具加载器

        加载器会在 discover() 时被调用。

        Args:
            loader: 加载函数
            category: 工具分类（应用于所有加载的工具）
            dependencies: 共享依赖声明
            requires_config: 是否需要 AIConfig 参数
            source: 加载器来源
        """
        if isinstance(category, str):
            category = ToolCategory(category)

        with self._lock:
            self._loaders.append(
                ToolLoaderSpec(
                    loader=loader,
                    category=category,
                    dependencies=dependencies or [],
                    requires_config=requires_config,
                    source=source,
                )
            )
            logger.debug(f"Registered tool loader from {source}")

    # -----------------------------------------------------------------------
    # 查询方法
    # -----------------------------------------------------------------------

    def get(self, name: str) -> Optional["BaseTool"]:
        """按名称获取工具

        Args:
            name: 工具名称

        Returns:
            对应的 BaseTool，未注册时返回 None
        """
        with self._lock:
            return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """获取工具元数据"""
        with self._lock:
            return self._metadata.get(name)

    def has(self, name: str) -> bool:
        """检查工具是否已注册"""
        with self._lock:
            return name in self._tools

    def list_tools(self) -> List["BaseTool"]:
        """列出所有已注册的工具"""
        with self._lock:
            return list(self._tools.values())

    def list_names(self) -> List[str]:
        """列出所有已注册的工具名称"""
        with self._lock:
            return list(self._tools.keys())

    def get_by_category(
        self,
        category: Union[ToolCategory, str],
    ) -> List["BaseTool"]:
        """按分类获取工具

        Args:
            category: 工具分类

        Returns:
            该分类下的所有工具列表
        """
        if isinstance(category, str):
            category = ToolCategory(category)

        with self._lock:
            return [
                self._tools[name]
                for name, meta in self._metadata.items()
                if meta.category == category
            ]

    def get_by_tag(self, tag: str) -> List["BaseTool"]:
        """按标签获取工具

        Args:
            tag: 标签

        Returns:
            包含该标签的所有工具列表
        """
        with self._lock:
            return [
                self._tools[name]
                for name, meta in self._metadata.items()
                if tag in meta.tags
            ]

    def get_categories_summary(self) -> Dict[str, int]:
        """获取各分类的工具数量统计"""
        with self._lock:
            summary: Dict[str, int] = {}
            for meta in self._metadata.values():
                cat = meta.category.value
                summary[cat] = summary.get(cat, 0) + 1
            return summary

    # -----------------------------------------------------------------------
    # 发现与加载
    # -----------------------------------------------------------------------

    def discover(
        self,
        config: Optional["AIConfig"] = None,
        *,
        force: bool = False,
    ) -> int:
        """发现并加载所有工具

        执行顺序：
        1. 加载内置工具（通过已注册的 loaders）
        2. 尝试加载外部工具（InnerAgentWorkflow/ai_tools）

        Args:
            config: AI 配置（大部分工具需要）
            force: 是否强制重新发现（默认只发现一次）

        Returns:
            新注册的工具数量
        """
        with self._lock:
            if self._discovered and not force:
                logger.debug("Tools already discovered, skipping")
                return 0

            count = 0

            # 如果没有传入 config，尝试获取
            # if config is None:
            try:
                from ..ai_config.ai_config import (
                    get_ai_config,
                )

                config = get_ai_config()
            except Exception as e:
                logger.warning(f"Failed to get AIConfig: {e}")

            # 1. 执行已注册的 loaders
            for spec in self._loaders:
                try:
                    if spec.requires_config:
                        if config is None:
                            logger.debug(
                                f"Skipping loader {spec.source}: requires config"
                            )
                            continue
                        tools = spec.loader(config)
                    else:
                        tools = spec.loader()

                    for tool in tools:
                        try:
                            self.register(
                                tool,
                                category=spec.category,
                                dependencies=spec.dependencies,
                                source=spec.source,
                                overwrite=False,
                            )
                            count += 1
                        except ValueError:
                            # 已注册，跳过
                            pass

                except Exception as e:
                    logger.error(f"Failed to load tools from {spec.source}: {e}")

            # 2. 尝试加载外部工具
            count += self._discover_external_tools(config)

            self._discovered = True
            logger.info(
                f"Tool discovery complete: {count} tools registered "
                f"({self.get_categories_summary()})"
            )
            return count

    def _discover_external_tools(self, config: Optional["AIConfig"]) -> int:
        """发现外部工具（来自 InnerAgentWorkflow/ai_tools）

        Returns:
            新注册的工具数量
        """
        count = 0

        try:
            # 尝试导入外部工具模块
            external_module = importlib.import_module("tools")

            # 查找 load_external_tools 函数
            load_fn = getattr(external_module, "load_external_tools", None)
            if load_fn is None:
                logger.debug("No load_external_tools found in ai_tools")
                return 0

            # 加载工具（这会触发 _init_tool_metadata）
            if config is not None:
                tools = load_fn(config)
            else:
                # 尝试无参数调用
                try:
                    tools = load_fn()
                except TypeError:
                    logger.debug("External loader requires config, skipping")
                    return 0

            # 加载后获取工具元数据（确保 _init_tool_metadata 已执行）
            tool_metadata = getattr(external_module, "TOOL_METADATA", {})

            for tool in tools:
                try:
                    # 获取工具特定的元数据
                    meta = tool_metadata.get(tool.name, {})
                    deps = meta.get("dependencies", [])
                    tags = meta.get("tags", set())

                    # 外部工具可以覆盖内置工具（支持私有工具迁移场景）
                    self.register(
                        tool,
                        category=ToolCategory.EXTERNAL,
                        dependencies=deps,
                        source="tools",
                        tags=tags,
                        overwrite=True,
                    )
                    count += 1
                    logger.debug(f"Loaded external tool: {tool.name}")
                except ValueError:
                    # 不应该发生（overwrite=True）
                    pass

        except ImportError:
            logger.debug("ai_tools not available")
        except Exception as e:
            logger.error(f"Failed to load external tools: {e}")

        return count

    # -----------------------------------------------------------------------
    # 管理方法
    # -----------------------------------------------------------------------

    def unregister(self, name: str) -> bool:
        """取消注册工具

        Args:
            name: 工具名称

        Returns:
            是否成功取消（工具不存在时返回 False）
        """
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                del self._metadata[name]
                logger.debug(f"Unregistered tool: {name}")
                return True
            return False

    def clear(self) -> None:
        """清空注册表（主要用于测试）"""
        with self._lock:
            self._tools.clear()
            self._metadata.clear()
            self._loaders.clear()
            self._discovered = False

    def reset_discovery(self) -> None:
        """重置发现状态（允许重新执行 discover）"""
        with self._lock:
            self._discovered = False


# ===========================================================================
# 单例访问
# ===========================================================================


def get_tool_registry() -> ToolRegistry:
    """获取工具注册表单例"""
    global _REGISTRY_INSTANCE

    if _REGISTRY_INSTANCE is None:
        with _REGISTRY_LOCK:
            if _REGISTRY_INSTANCE is None:
                _REGISTRY_INSTANCE = ToolRegistry()

    return _REGISTRY_INSTANCE


# ===========================================================================
# 便捷工具定义装饰器
# ===========================================================================


def register_tool(
    *,
    category: Union[ToolCategory, str] = ToolCategory.OTHER,
    dependencies: Optional[List[ToolDependency]] = None,
    tags: Optional[Set[str]] = None,
) -> Callable[[Type], Type]:
    """工具类装饰器（用于声明式注册）

    使用示例:
        @register_tool(
            category=ToolCategory.MEDIA,
            dependencies=[
                ToolDependency(DependencyType.MEDIA_REGISTRY),
                ToolDependency(DependencyType.CONFIG_PROVIDER, provider="doubao"),
            ],
            tags={"detection", "vlm"}
        )
        class DetectionToolLoader:
            @staticmethod
            def load(config: AIConfig) -> List[StructuredTool]:
                # ... 加载逻辑
                pass

    注意：这个装饰器主要用于声明元数据，实际注册仍需调用 register_loader。
    """

    def decorator(cls: Type) -> Type:
        # 存储元数据到类属性
        cls._tool_category = (
            ToolCategory(category) if isinstance(category, str) else category
        )
        cls._tool_dependencies = dependencies or []
        cls._tool_tags = tags or set()
        return cls

    return decorator


# ===========================================================================
# 依赖检查工具
# ===========================================================================


def check_dependencies(
    metadata: ToolMetadata,
    config: Optional["AIConfig"] = None,
) -> List[str]:
    """检查工具依赖是否满足

    Args:
        metadata: 工具元数据
        config: AI 配置

    Returns:
        未满足的依赖描述列表（空列表表示全部满足）
    """
    missing: List[str] = []

    for dep in metadata.dependencies:
        if not dep.required:
            continue

        satisfied = False

        if dep.type == DependencyType.CONFIG_PROVIDER:
            if config and dep.provider:
                satisfied = dep.provider in config.providers
                if satisfied:
                    provider = config.providers[dep.provider]
                    satisfied = bool(provider.api_key and provider.base_url)

        elif dep.type == DependencyType.CONFIG_MEDIA:
            # CONFIG_MEDIA已废弃，媒体配置已分散到各模块
            satisfied = config is not None

        elif dep.type == DependencyType.CONFIG_TTS:
            satisfied = config is not None and config.tts is not None

        elif dep.type == DependencyType.CONFIG_MUSIC:
            satisfied = config is not None and config.music is not None

        elif dep.type == DependencyType.MEDIA_REGISTRY:
            try:
                from ..ai_media_resource import (
                    get_media_registry,
                )

                get_media_registry()
                satisfied = True
            except Exception:
                satisfied = False

        elif dep.type == DependencyType.STORAGE_ADAPTER:
            try:
                from ..ai_media_resource import (
                    get_storage_adapter,
                )

                get_storage_adapter()
                satisfied = True
            except Exception:
                satisfied = False

        elif dep.type == DependencyType.SCENE_SERVICE:
            try:
                from Backend.utils import get_scene_service

                get_scene_service()
                satisfied = True
            except Exception:
                satisfied = False

        else:
            # 其他依赖类型暂时跳过检查
            satisfied = True

        if not satisfied:
            missing.append(str(dep))

    return missing


__all__ = [
    # 枚举和数据类
    "ToolCategory",
    "DependencyType",
    "ToolDependency",
    "ToolMetadata",
    "ToolLoaderSpec",
    # 注册表
    "ToolRegistry",
    "get_tool_registry",
    # 装饰器和工具函数
    "register_tool",
    "check_dependencies",
]
