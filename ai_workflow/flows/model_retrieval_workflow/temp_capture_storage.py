from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _is_ascii_path(path: Path) -> bool:
    return str(path).isascii()


def _is_dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_file = path / ".write_probe"
        with open(probe_file, "w", encoding="ascii") as fp:
            fp.write("ok")
        probe_file.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def build_temp_capture_root() -> Path:
    """构建可控且可写的 ASCII 临时截图目录。

    固定使用系统临时目录下的子目录：<tempdir>/corona_capture_tmp
    若系统临时目录不是 ASCII 或不可写，直接抛出异常。
    """
    temp_root = Path(tempfile.gettempdir()) / "corona_capture_tmp"

    if not _is_ascii_path(temp_root):
        raise RuntimeError(f"系统临时目录不是 ASCII 路径: {temp_root}")

    if _is_dir_writable(temp_root):
        return temp_root

    raise RuntimeError(
        f"无法创建系统临时截图目录: {temp_root}"
    )


def _sanitize_ascii_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return "object"

    sanitized_chars = []
    for char in text:
        if char.isascii() and (char.isalnum() or char in {"-", "_"}):
            sanitized_chars.append(char)
        elif char.isspace():
            sanitized_chars.append("_")
        else:
            sanitized_chars.append("_")

    sanitized = "".join(sanitized_chars).strip("_")
    return sanitized or "object"


def _is_written_image(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def make_temp_capture_path(temp_root: Path, actor_name: str, view_name: str) -> Path:
    """为截图生成 ASCII 临时文件路径。"""
    safe_actor_name = _sanitize_ascii_name(actor_name)
    return temp_root / f"{safe_actor_name}_{view_name}.png"


def save_to_temp_then_move(
    camera: Any,
    *,
    temp_path: Path,
    final_path: str,
    actor_name: str,
    view_name: str,
) -> str:
    """先保存到 ASCII 临时路径，再由 Python 移动到目标路径。"""
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    final_dir = os.path.dirname(final_path)
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)

    temp_path_str = str(temp_path)

    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass

    try:
        if os.path.exists(final_path):
            os.remove(final_path)
    except OSError:
        pass

    camera.save_screenshot_sync(temp_path_str)
    if not _is_written_image(temp_path_str):
        logger.error(
            "[Workflow][capture] 临时截图写入失败: actor=%s, view=%s, path=%s",
            actor_name,
            view_name,
            temp_path_str,
        )
        return ""

    try:
        shutil.move(temp_path_str, final_path)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[Workflow][capture] 截图移动失败: actor=%s, view=%s, temp=%s, final=%s, err=%s",
            actor_name,
            view_name,
            temp_path_str,
            final_path,
            exc,
        )
        return ""

    if not _is_written_image(final_path):
        logger.error(
            "[Workflow][capture] 截图目标文件校验失败: actor=%s, view=%s, path=%s",
            actor_name,
            view_name,
            final_path,
        )
        return ""

    return final_path
