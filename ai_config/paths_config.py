"""
AITool 路径配置入口。

优先透传 editor/CabbageEditor/config/paths_config.py 中的实现；
若该文件不存在，则使用 AITool 内部的后备实现。
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)


def _editor_paths_config_path() -> Path:
    """返回编辑器侧 paths_config.py 的路径。"""
    return Path(__file__).resolve().parents[4] / "config" / "paths_config.py"


def _load_editor_paths_module() -> ModuleType | None:
    """按文件路径加载编辑器侧 paths_config 模块。"""
    config_path = _editor_paths_config_path()
    if not config_path.is_file():
        return None

    try:
        spec = importlib.util.spec_from_file_location(
            "cabbageeditor_editor_paths_config",
            config_path,
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        logger.warning("加载编辑器路径配置失败，回退到 AITool 默认实现: %s", exc)
        return None


_EDITOR_PATHS_MODULE = _load_editor_paths_module()


@dataclass(frozen=True)
class PathsConfig:
    """路径配置。"""

    repo_root: Path
    backend_root: Path
    frontend_dist: Path
    script_dir: Path
    autosave_dir: Path
    config_dir: Path
    assets_model_dir: Path
    object_recognition_db: Path
    screenshots_dir: Optional[Path] = None
    media_local_storage: Optional[Path] = None


def _get_aitool_repo_root() -> Path:
    """获取 CoronaEngine 仓库根目录。"""
    return Path(__file__).resolve().parents[6]


def _get_editor_root() -> Path:
    """获取 CabbageEditor 根目录。"""
    return Path(__file__).resolve().parents[4]


def get_project_root() -> Path:
    """兼容旧接口：获取 CoronaEngine 仓库根目录。"""
    return _get_aitool_repo_root()


def get_aitool_root() -> Path:
    """兼容旧接口：获取 AITool 插件根目录。"""
    return Path(__file__).resolve().parents[1]


def get_default_paths() -> PathsConfig:
    """获取默认路径配置。"""
    if _EDITOR_PATHS_MODULE is not None and hasattr(_EDITOR_PATHS_MODULE, "get_default_paths"):
        return _EDITOR_PATHS_MODULE.get_default_paths()

    repo_root = _get_aitool_repo_root()
    editor_root = _get_editor_root()
    config_dir = editor_root / "config"
    assets_model_dir = get_project_models_dir()
    autosave_dir = get_project_media_dir()

    return PathsConfig(
        repo_root=repo_root,
        backend_root=repo_root / "Backend",
        frontend_dist=repo_root / "Frontend" / "dist" / "index.html",
        script_dir=repo_root / "Backend" / "script",
        autosave_dir=autosave_dir,
        config_dir=config_dir,
        assets_model_dir=assets_model_dir,
        object_recognition_db=get_project_recognition_db(),
        screenshots_dir=get_project_screenshots_dir(),
        media_local_storage=autosave_dir,
    )


def _get_active_project_path() -> Path:
    """获取当前活跃项目路径，未打开项目时回退到 cwd。"""
    if _EDITOR_PATHS_MODULE is not None and hasattr(_EDITOR_PATHS_MODULE, "_get_active_project_path"):
        return _EDITOR_PATHS_MODULE._get_active_project_path()

    try:
        from CoronaCore.core.corona_editor import CoronaEditor

        project_path = CoronaEditor.CoronaEngine.active_project_path
        if project_path:
            return Path(project_path)
    except Exception:
        pass

    return Path(os.getcwd())


def get_project_media_dir() -> Path:
    """获取当前项目的媒体存储目录。"""
    if _EDITOR_PATHS_MODULE is not None and hasattr(_EDITOR_PATHS_MODULE, "get_project_media_dir"):
        return _EDITOR_PATHS_MODULE.get_project_media_dir()

    directory = _get_active_project_path() / "media"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_models_dir() -> Path:
    """获取当前项目的模型目录。"""
    if _EDITOR_PATHS_MODULE is not None and hasattr(_EDITOR_PATHS_MODULE, "get_project_models_dir"):
        return _EDITOR_PATHS_MODULE.get_project_models_dir()

    directory = _get_active_project_path() / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_screenshots_dir() -> Path:
    """获取当前项目的截图目录。"""
    if _EDITOR_PATHS_MODULE is not None and hasattr(_EDITOR_PATHS_MODULE, "get_project_screenshots_dir"):
        return _EDITOR_PATHS_MODULE.get_project_screenshots_dir()

    directory = _get_active_project_path() / "screenshots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_project_recognition_db() -> Path:
    """获取当前项目的物体识别数据库路径。"""
    if _EDITOR_PATHS_MODULE is not None and hasattr(_EDITOR_PATHS_MODULE, "get_project_recognition_db"):
        return _EDITOR_PATHS_MODULE.get_project_recognition_db()

    return get_project_models_dir() / "database.db"


__all__ = [
    "PathsConfig",
    "get_project_root",
    "get_aitool_root",
    "get_default_paths",
    "_get_active_project_path",
    "get_project_media_dir",
    "get_project_models_dir",
    "get_project_screenshots_dir",
    "get_project_recognition_db",
]
