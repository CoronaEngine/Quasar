import functools
import logging
import os

import importlib
import sys
import threading
from typing import Callable

import yaml

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
            modules_path = os.path.join(project_dir, "ai_modules")

            config_path = os.path.join(
                project_dir, "ai_service", "module_settings.yaml"
            )
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 解析模块配置
            if "modules" in config:
                for module_data in config["modules"]:
                    if not module_data.get("enabled", False):
                        logger.debug(f"跳过禁用模块: {module_data.get('name', '')}")
                        return
                    module_name = module_data.get("name", "")
                    module_dir = os.path.join(modules_path, module_name)

                    # 尝试导入 configs/settings.py
                    settings_path = os.path.join(module_dir, "configs", "settings.py")
                    if os.path.exists(settings_path):
                        try:
                            module_path = f"ai_modules.{module_name}.configs.settings"
                            importlib.import_module(module_path)
                            logger.info(f"✓ 成功导入配置模块: {module_name}")
                        except Exception as e:
                            logger.error(f"✗ 导入配置模块失败 {module_name}: {e}")

                    # 尝试导入 base.py
                    base_path = os.path.join(module_dir, "base.py")
                    if os.path.exists(base_path):
                        try:
                            module_path = f"ai_modules.{module_name}.base"
                            importlib.import_module(module_path)
                            logger.info(f"✓ 成功导入基础模块: {module_name}")
                        except Exception as e:
                            logger.error(f"✗ 导入基础模块失败 {module_name}: {e}")

                    # 尝试导入 loader.py
                    loader_path = os.path.join(module_dir, "tools", "loader.py")
                    if os.path.exists(loader_path):
                        try:
                            module_path = f"ai_modules.{module_name}.tools.loader"
                            importlib.import_module(module_path)
                            logger.info(f"✓ 成功导入loader模块: {module_name}")
                        except Exception as e:
                            logger.error(f"✗ 导入loader模块失败 {module_name}: {e}")
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
        return wrapper

    return decorator


def get_ai_entrance():
    from ai_service import entrance

    _AI_ENTRANCE_CLASS = entrance.ai_entrance
    if not _AI_ENTRANCE_CLASS.if_import:
        _AI_ENTRANCE_CLASS.reimport()
    return _AI_ENTRANCE_CLASS
