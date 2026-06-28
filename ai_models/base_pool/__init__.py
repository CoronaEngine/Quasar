"""
客户端池模块

提供按类别管理的媒体生成客户端，支持两种模式：

1. 池化模式（检测到 ai_pool）：
   - 多账号加权轮询
   - 熔断与自动恢复
   - 多供应商适配器

2. 降级模式（未检测到账号池模块）：
   - 使用旧客户端单例 (client_dmx.py, client_speech.py, 等)
   - 从 AIConfig 自动加载配置
   - 提供相同的接口，对工具层透明

架构层次：
- MediaCategory: 媒体类别 (IMAGE/VIDEO/MUSIC/SPEECH/CHAT)
- MediaRequest/ImageRequest/...: 标准请求格式
- MediaResult/MultiMediaResult: 标准响应格式
- PoolRegistry: 智能注册表，自动选择池或降级

使用示例：
    from models.pool import (
        get_pool_registry,
        MediaCategory,
        ImageRequest,
    )

    # 初始化（应用启动时，可选）
    initialize_account_pools()

    # 创建任务（自动选择池或降级模式）
    request = ImageRequest(session_id="xxx", prompt="一只猫")
    task = get_pool_registry().create_task(MediaCategory.IMAGE, request)
    if task:
        result = task()  # 返回 MediaResult
"""

# 类别
from .category import (
    MediaCategory,
    CATEGORY_CONTENT_TYPE,
    get_content_type,
)

# 请求
from .requests import (
    MediaRequest,
    ImageRequest,
    VideoRequest,
    MusicRequest,
    SpeechRequest,
    ChatRequest,
    OmniRequest,
    DetectionRequest,
)

# 响应
from .responses import (
    MediaResult,
    MultiMediaResult,
    ChatResult,
    to_storage_result,
    build_result_part,
)

# 注册表与初始化
from .registry import (
    PoolRegistry,
    get_pool_registry,
    reset_pool_registry,
    get_category_pool,
    create_media_task,
    initialize_account_pools,
    is_pool_initialized,
    reset_pool_initialization,
    is_pool_mode,
    reset_pool_detection,
    get_chat_model,
)

# 降级适配器（内部使用，但导出供测试）
from .legacy_fallback import (
    get_legacy_client,
    clear_legacy_clients,
    create_legacy_task,
)

__all__ = [
    # 类别
    "MediaCategory",
    "CATEGORY_CONTENT_TYPE",
    "get_content_type",
    # 请求
    "MediaRequest",
    "ImageRequest",
    "VideoRequest",
    "MusicRequest",
    "SpeechRequest",
    "ChatRequest",
    "OmniRequest",
    "DetectionRequest",
    # 响应
    "MediaResult",
    "MultiMediaResult",
    "ChatResult",
    "to_storage_result",
    "build_result_part",
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
    # 降级适配器
    "get_legacy_client",
    "clear_legacy_clients",
    "create_legacy_task",
]
