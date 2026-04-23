"""
工具加载模块

统一加载所有可用工具，通过 ToolRegistry 管理：
1. 内置工具（omni/text/scene）
2. 外部工具（InnerAgentWorkflow/ai_tools）

工具分类：
- test: 测试工具（无依赖）
- text: 文案生成工具（依赖 LLM）
- scene: 场景操作工具（依赖 scene_service）
- omni: 媒体生成工具（图像、视频、语音、音乐、多模态理解）

使用方式：
    from tools import load_tools

    # 加载所有工具（同时注册到 ToolRegistry）
    tools = load_tools(config)

    # 或通过 ToolRegistry 获取
    from tools.registry import get_tool_registry
    registry = get_tool_registry()
    registry.discover(config)
    tools = registry.list_tools()
"""

from __future__ import annotations

import logging
from typing import Callable, List

from langchain_core.tools import BaseTool

from ai_config.ai_config import AIConfig
from ai_tools.registry import (
    ToolRegistry,
    ToolCategory,
    ToolDependency,
    DependencyType,
    get_tool_registry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 宣主可扩展的内置 loader 注册钩子
# ---------------------------------------------------------------------------

_EXTRA_BUILTIN_REGISTRARS: List[Callable[[ToolRegistry], None]] = []


def register_extra_builtin_registrar(
    registrar: Callable[[ToolRegistry], None],
) -> None:
    """注入额外的内置 loader 注册函数。

    宣主（如编辑器）可以在启动时调用本函数注册额外的引擎相关 loader。
    注册函数会在首次的 :func:`load_tools` 中被调用一次。
    """
    if registrar in _EXTRA_BUILTIN_REGISTRARS:
        return
    _EXTRA_BUILTIN_REGISTRARS.append(registrar)


# ===========================================================================
# 内置工具加载器注册
# ===========================================================================


def _register_builtin_loaders(registry: ToolRegistry) -> None:
    """注册所有内置工具加载器

    每个加载器声明其：
    - category: 工具分类
    - dependencies: 依赖声明
    - source: 来源模块

    注意：此函数只注册加载器，不执行加载。
    实际加载在 registry.discover() 时进行。
    """
    # -----------------------------------------------------------------------
    # 测试工具（无依赖）
    # -----------------------------------------------------------------------
    from ai_tools.test import load_test_tools

    registry.register_loader(
        loader=load_test_tools,
        category=ToolCategory.TEST,
        dependencies=[],
        requires_config=False,
        source="tools.test",
    )

    # -----------------------------------------------------------------------
    # 文案生成工具
    # -----------------------------------------------------------------------
    from ai_modules.text_generate.tools.text_tools import load_text_tools

    registry.register_loader(
        loader=load_text_tools,
        category=ToolCategory.TEXT,
        dependencies=[
            ToolDependency(DependencyType.LLM, required=True),
            ToolDependency(
                DependencyType.CONFIG_PROVIDER, provider="doubao", required=True
            ),
        ],
        requires_config=True,
        source="tools.text_tools",
    )

    # -----------------------------------------------------------------------
    # 场景操作类、摄像头、模型导入等引擎相关工具已迁出 CAI；由宣主侧
    # 通过 :func:`register_extra_builtin_registrar` 补齐。

    # -----------------------------------------------------------------------
    # 图像生成工具
    # -----------------------------------------------------------------------
    from ai_modules.image_generate.tools.image_tools import load_image_tools

    registry.register_loader(
        loader=load_image_tools,
        category=ToolCategory.MEDIA,
        dependencies=[
            ToolDependency(DependencyType.CONFIG_MEDIA, required=True),
            ToolDependency(DependencyType.MEDIA_REGISTRY, required=True),
            ToolDependency(DependencyType.STORAGE_ADAPTER, required=True),
            ToolDependency(DependencyType.HTTP_IMAGE, required=True),
        ],
        requires_config=True,
        source="tools.omni.image_tools",
    )

    # -----------------------------------------------------------------------
    # 视频生成工具
    # -----------------------------------------------------------------------
    from ai_modules.video_generate.tools.video_tools import load_video_tools

    registry.register_loader(
        loader=load_video_tools,
        category=ToolCategory.MEDIA,
        dependencies=[
            ToolDependency(DependencyType.CONFIG_MEDIA, required=True),
            ToolDependency(DependencyType.MEDIA_REGISTRY, required=True),
            ToolDependency(DependencyType.STORAGE_ADAPTER, required=True),
            ToolDependency(DependencyType.HTTP_VIDEO, required=True),
        ],
        requires_config=True,
        source="tools.omni.video_tools",
    )

    # -----------------------------------------------------------------------
    # 语音合成工具
    # -----------------------------------------------------------------------
    from ai_modules.speech_generate.tools.speech_tools import load_speech_tools

    registry.register_loader(
        loader=load_speech_tools,
        category=ToolCategory.MEDIA,
        dependencies=[
            ToolDependency(DependencyType.CONFIG_TTS, required=True),
            ToolDependency(DependencyType.MEDIA_REGISTRY, required=True),
            ToolDependency(DependencyType.STORAGE_ADAPTER, required=True),
            ToolDependency(DependencyType.HTTP_TTS, required=True),
        ],
        requires_config=True,
        source="tools.omni.speech_tools",
    )

    # -----------------------------------------------------------------------
    # 音乐生成工具
    # -----------------------------------------------------------------------
    from ai_modules.music_generate.tools.music_tools import load_music_tools

    registry.register_loader(
        loader=load_music_tools,
        category=ToolCategory.MEDIA,
        dependencies=[
            ToolDependency(DependencyType.CONFIG_MUSIC, required=True),
            ToolDependency(DependencyType.MEDIA_REGISTRY, required=True),
            ToolDependency(DependencyType.STORAGE_ADAPTER, required=True),
            ToolDependency(DependencyType.HTTP_MUSIC, required=True),
        ],
        requires_config=True,
        source="tools.omni.music_tools",
    )

    # -----------------------------------------------------------------------
    # 多模态理解工具
    # -----------------------------------------------------------------------
    from ai_modules.omni.tools.omni_tools import load_omni_tools

    registry.register_loader(
        loader=load_omni_tools,
        category=ToolCategory.MEDIA,
        dependencies=[
            ToolDependency(DependencyType.CONFIG_MEDIA, required=True),
            ToolDependency(DependencyType.VLM, required=True),
            ToolDependency(DependencyType.MEDIA_REGISTRY, required=True),
        ],
        requires_config=True,
        source="tools.omni.omni_tools",
    )

# -----------------------------------------------------------------------
# 3D 生成工具（Rodin）
# -----------------------------------------------------------------------
    from ai_modules.three_d_generate.tools.model_tools import load_3d_tools

    registry.register_loader(
        loader=load_3d_tools,
        category=ToolCategory.SCENE,
        dependencies=[
            # ToolDependency(DependencyType.CONFIG_PROVIDER, provider="rodin"),
        ],
        requires_config=True,
        source="tools.model_tools",
    )

# -----------------------------------------------------------------------
# 3D 生成工具（混元3D）
# -----------------------------------------------------------------------
    from ai_modules.three_d_generate.tools.model_tools import load_hunyuan3d_tools

    registry.register_loader(
        loader=load_hunyuan3d_tools,
        category=ToolCategory.SCENE,
        dependencies=[],
        requires_config=True,
        source="tools.hunyuan3d_tools",
    )

# -----------------------------------------------------------------------
#  物体识别工具（Qwen3-VL-Embedding + sqlite-vec）
# -----------------------------------------------------------------------
    from ai_modules.object_recognition.base import load_recognition_tools

    registry.register_loader(
        loader=load_recognition_tools,
        category=ToolCategory.SCENE,
        dependencies=[],
        requires_config=True,
        source="tools.base",
    )

# -----------------------------------------------------------------------
#  场景拆解工具（breakdown）
# -----------------------------------------------------------------------
    try:
        from ai_modules.scene_breakdown.tools.scene_breakdown_tools import load_scene_breakdown_tools

        registry.register_loader(
            loader=load_scene_breakdown_tools,
            category=ToolCategory.SCENE,
            dependencies=[
                # ToolDependency(DependencyType.CONFIG_PROVIDER, provider="rodin"),
            ],
            requires_config=True,
            source="tools.scene_breakdown_tools",
        )
    except ImportError as e:
        logger.warning("scene_breakdown tools not available (skipped): %s", e)

    # -----------------------------------------------------------------------
    #  场景布局、场景审查等引擎相关工具已迁出 CAI，由宣主侧插入。
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 执行宣主注入的额外 loader 注册函数
    # -----------------------------------------------------------------------
    for fn in list(_EXTRA_BUILTIN_REGISTRARS):
        try:
            fn(registry)
        except Exception as exc:
            logger.exception("额外 builtin loader 注册函数执行失败: %s", exc)


# ===========================================================================
# 公开 API
# ===========================================================================


def load_tools(config: AIConfig) -> List[BaseTool]:
    """
    加载所有可用工具

    执行顺序：
    1. 注册内置工具加载器
    2. 执行发现（加载内置 + 外部工具）
    3. 返回所有已注册工具

    参数:
    - config: AI 配置

    返回:
    - BaseTool 列表
    """
    registry = get_tool_registry()

    # 注册内置加载器（仅首次调用时生效，后续调用自动跳过）
    if not registry._loaders:
        _register_builtin_loaders(registry)

    # 执行发现（加载内置 + 外部工具）
    registry.discover(config)

    # 返回所有工具
    return registry.list_tools()


def get_tools_by_category(category: str) -> List[BaseTool]:
    """按分类获取工具（需先调用 load_tools）"""
    return get_tool_registry().get_by_category(category)


def get_tool_by_name(name: str) -> BaseTool | None:
    """按名称获取工具（需先调用 load_tools）"""
    return get_tool_registry().get(name)


__all__ = [
    "load_tools",
    "get_tools_by_category",
    "get_tool_by_name",
]
