import functools
import logging
import os

import sys
import threading
from typing import Callable

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from ai_tools.ai_config_collector import ConfigCollector

logger = logging.getLogger(__name__)


def _log_runtime_paths() -> None:
    """在 AI 模块完成启动后输出路径解析结果。"""
    try:
        from ai_config.paths_config import (
            _get_active_project_path,
            get_default_paths,
            get_project_media_dir,
            get_project_models_dir,
            get_project_screenshots_dir,
            get_project_recognition_db,
        )

        paths = get_default_paths()
        logger.info("[AITool Paths] ===== AI modules startup path summary =====")
        logger.info(
            "[AITool Paths] active_project_path=%s",
            _get_active_project_path().resolve(),
        )
        logger.info("[AITool Paths] repo_root=%s", paths.repo_root)
        logger.info("[AITool Paths] backend_root=%s", paths.backend_root)
        logger.info("[AITool Paths] frontend_dist=%s", paths.frontend_dist)
        logger.info("[AITool Paths] script_dir=%s", paths.script_dir)
        logger.info("[AITool Paths] config_dir=%s", paths.config_dir)
        logger.info("[AITool Paths] autosave_dir=%s", paths.autosave_dir)
        logger.info("[AITool Paths] media_local_storage=%s", paths.media_local_storage)
        logger.info("[AITool Paths] assets_model_dir=%s", paths.assets_model_dir)
        logger.info(
            "[AITool Paths] object_recognition_db=%s", paths.object_recognition_db
        )
        logger.info("[AITool Paths] screenshots_dir=%s", get_project_screenshots_dir())
        logger.info("[AITool Paths] project_media_dir=%s", get_project_media_dir())
        logger.info("[AITool Paths] project_models_dir=%s", get_project_models_dir())
        logger.info(
            "[AITool Paths] project_recognition_db=%s", get_project_recognition_db()
        )
    except Exception as exc:
        logger.warning("[AITool Paths] startup path summary failed: %s", exc)


class ai_entrance:
    collector = ConfigCollector()
    if_import = False
    _lock = threading.Lock()

    @classmethod
    def reimport(cls):
        with cls._lock:
            if cls.if_import:
                return
            from cai import get_default_runtime

            runtime = get_default_runtime()
            loaded = runtime.plugin_manager.load_module_settings(
                os.path.join(project_dir, "ai_service", "module_settings.yaml"),
                os.path.join(project_dir, "ai_modules"),
            )
            logger.info(
                "ai_modules imported: configs=%d base=%d loader=%d failed=%d",
                len(loaded["configs"]),
                len(loaded["base"]),
                len(loaded["loader"]),
                len(loaded["failed"]),
            )
            logger.debug("ai_modules detail: %s", loaded)
            _log_runtime_paths()
            ai_entrance.if_import = True


def register_entrance(handler_name: str = None):
    """
    将函数注册为 ai_entrance 的静态方法

    Args:
        handler_name: 在 ai_entrance 中的方法名，默认使用原函数名
    """

    def decorator(func: Callable) -> Callable:
        # 确定在 ai_entrance 中的方法名
        method_name = handler_name or func.__name__

        # 创建包装函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(ai_entrance, method_name, staticmethod(wrapper))
        try:
            from cai import get_default_runtime

            get_default_runtime().register_entrance_handler(method_name, wrapper)
        except Exception as exc:
            logger.debug("runtime entrance handler register skipped for %s: %s", method_name, exc)
        return wrapper

    return decorator


def get_ai_entrance():
    from cai import get_default_runtime

    return get_default_runtime().get_ai_entrance()
