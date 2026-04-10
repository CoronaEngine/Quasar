"""物体识别模块初始化。"""

import logging

logger = logging.getLogger(__name__)


def _log_module_ready() -> None:
    """记录模块已启用云端 embedding 模式。"""
    logger.info("物体识别模块初始化完成，当前仅支持云端 embedding 服务")


_log_module_ready()
