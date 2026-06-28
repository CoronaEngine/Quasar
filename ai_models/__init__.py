# """
# Models 模块
# 提供 AI 模型加载器和通用工具
#
# 子模块：
# - pool: 客户端池抽象层，自动检测账号池或降级到旧客户端
# - chat_loader: LLM 聊天模型加载器（旧接口，现由 pool 统一提供）
# - client_*: 各类媒体生成旧客户端（单例模式）
# """
#
# # 旧客户端（供降级模式和直接调用使用）
# from ai_modules.video_generate.tools.client_video import DashScopeVideoClient
# from ai_modules.speech_generate.tools.client_speech import TTSClient
# from ai_modules.music_generate.tools.client_music import SunoMusicClient
#
# # pool 子模块的便捷导出
# # get_chat_model 现在由 pool 模块提供，自动检测池模式或降级模式
# from ai_models.base_pool import (
#     get_pool_registry,
#     initialize_account_pools,
#     is_pool_initialized,
#     is_pool_mode,
#     get_chat_model,
# )
#
# # 保留旧的 chat_loader 入口用于直接调用（不经过池系统）
# from ai_modules.text_generate.tools.chat_loader import get_chat_model as get_chat_model_legacy
#
# __all__ = [
#     # 模型加载（统一入口，推荐使用）
#     "get_chat_model",
#     # 旧的直接加载方式（不经过池系统）
#     "get_chat_model_legacy",
#     # 旧客户端（供降级模式使用）
#     "DashScopeVideoClient",
#     "TTSClient",
#     "SunoMusicClient",
#     # 池系统便捷访问
#     "get_pool_registry",
#     "initialize_account_pools",
#     "is_pool_initialized",
#     "is_pool_mode",
# ]
