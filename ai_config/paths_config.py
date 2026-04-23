"""
AITool 路径配置入口（通用 AI 库版本）。

设计：
- CAI 不再硬依赖任何宿主（编辑器/引擎）；
- 宿主可通过 :func:`set_paths_resolver` 注入自定义解析器；
- 未注入时使用基于环境变量 / 当前工作目录的默认实现。

环境变量：
- ``CAI_PROJECT_ROOT``：当前活跃项目根目录，未设置时回退到 ``os.getcwd()``。
- ``CAI_REPO_ROOT``：仓库根目录，未设置时回退到 CAI 包所在父目录。
- ``CAI_MEDIA_DIR`` / ``CAI_MODELS_DIR`` / ``CAI_SCREENSHOTS_DIR`` /
  ``CAI_RECOGNITION_DB`` / ``CAI_CONFIG_DIR``：可选的精确覆盖。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# 宿主可注入的解析器
# ---------------------------------------------------------------------------


class PathsResolver:
    """路径解析器协议（duck typing）。

    宿主侧可以提供一个对象，按需实现以下方法（任意可省略，省略时回退到默认实现）：

    - ``get_active_project_path() -> Path``
    - ``get_project_media_dir() -> Path``
    - ``get_project_models_dir() -> Path``
    - ``get_project_screenshots_dir() -> Path``
    - ``get_project_recognition_db() -> Path``
    - ``get_default_paths() -> PathsConfig``
    """


_resolver: Optional[object] = None


def set_paths_resolver(resolver: Optional[object]) -> None:
    """注入宿主侧路径解析器。传入 ``None`` 可清除。"""
    global _resolver
    _resolver = resolver


def get_paths_resolver() -> Optional[object]:
    """获取当前已注入的解析器（可能为 None）。"""
    return _resolver


def _resolver_call(method_name: str) -> Optional[object]:
    """若已注入解析器且实现了指定方法，则调用并返回结果；否则返回 None。"""
    resolver = _resolver
    if resolver is None:
        return None
    fn: Optional[Callable[[], object]] = getattr(resolver, method_name, None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception as exc:
        logger.warning("paths_resolver.%s 执行失败，回退默认实现: %s", method_name, exc)
        return None


# ---------------------------------------------------------------------------
# 默认实现
# ---------------------------------------------------------------------------


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value) if value else None


def _get_aitool_repo_root() -> Path:
    """获取 CAI 仓库根目录（默认实现）。"""
    return _env_path("CAI_REPO_ROOT") or Path(__file__).resolve().parents[1]


def _get_active_project_path() -> Path:
    """获取当前活跃项目路径（默认实现）。

    保留下划线前缀名以兼容历史调用方。
    """
    result = _resolver_call("get_active_project_path")
    if isinstance(result, Path):
        return result
    if isinstance(result, str):
        return Path(result)

    return _env_path("CAI_PROJECT_ROOT") or Path(os.getcwd())


def _ensure_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.debug("创建目录失败 %s: %s", path, exc)
    return path


def get_project_root() -> Path:
    """兼容旧接口：获取仓库根目录。"""
    return _get_aitool_repo_root()


def get_aitool_root() -> Path:
    """兼容旧接口：获取 AITool / CAI 包根目录。"""
    return Path(__file__).resolve().parents[1]


def get_project_media_dir() -> Path:
    """获取当前项目的媒体存储目录。"""
    result = _resolver_call("get_project_media_dir")
    if isinstance(result, Path):
        return result
    if isinstance(result, str):
        return Path(result)

    override = _env_path("CAI_MEDIA_DIR")
    return _ensure_dir(override or (_get_active_project_path() / "media"))


def get_project_models_dir() -> Path:
    """获取当前项目的模型目录。"""
    result = _resolver_call("get_project_models_dir")
    if isinstance(result, Path):
        return result
    if isinstance(result, str):
        return Path(result)

    override = _env_path("CAI_MODELS_DIR")
    return _ensure_dir(override or (_get_active_project_path() / "models"))


def get_project_screenshots_dir() -> Path:
    """获取当前项目的截图目录。"""
    result = _resolver_call("get_project_screenshots_dir")
    if isinstance(result, Path):
        return result
    if isinstance(result, str):
        return Path(result)

    override = _env_path("CAI_SCREENSHOTS_DIR")
    return _ensure_dir(override or (_get_active_project_path() / "screenshots"))


def get_project_recognition_db() -> Path:
    """获取当前项目的物体识别数据库路径。"""
    result = _resolver_call("get_project_recognition_db")
    if isinstance(result, Path):
        return result
    if isinstance(result, str):
        return Path(result)

    override = _env_path("CAI_RECOGNITION_DB")
    return override or (get_project_models_dir() / "database.db")


def get_default_paths() -> PathsConfig:
    """获取默认路径配置。"""
    result = _resolver_call("get_default_paths")
    if isinstance(result, PathsConfig):
        return result

    repo_root = _get_aitool_repo_root()
    config_dir = _env_path("CAI_CONFIG_DIR") or (repo_root / "config")
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


__all__ = [
    "PathsConfig",
    "PathsResolver",
    "set_paths_resolver",
    "get_paths_resolver",
    "get_project_root",
    "get_aitool_root",
    "get_default_paths",
    "_get_active_project_path",
    "get_project_media_dir",
    "get_project_models_dir",
    "get_project_screenshots_dir",
    "get_project_recognition_db",
]
