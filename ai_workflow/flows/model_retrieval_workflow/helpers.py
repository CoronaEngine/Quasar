from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ai_config.ai_config import get_ai_config
from ai_tools.registry import get_tool_registry
from ai_tools.response_adapter import FILEID_SCHEME
from ai_config.paths_config import _get_active_project_path

logger = logging.getLogger(__name__)

PREVIEW_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _resolve_preview_part_url(part: Dict[str, Any]) -> str:
    """解析 3D 工具 image part，优先返回可展示的预览图路径。"""
    raw_url = str(part.get("content_url") or "").strip()
    if raw_url.startswith(FILEID_SCHEME):
        file_id = raw_url[len(FILEID_SCHEME):].strip()
        if file_id:
            try:
                from ai_media_resource import get_media_registry

                resolved = str(get_media_registry().resolve(file_id) or "").strip()
                if resolved:
                    return resolved
            except Exception as exc:  # noqa: BLE001
                logger.warning("3D 预览图 file_id 解析失败: %s, err=%s", file_id, exc)

    for key in ("content_url", "content_path", "content_text"):
        candidate = str(part.get(key) or "").strip()
        if not candidate:
            continue

        lowered = candidate.lower()
        if lowered.startswith(("http://", "https://", "data:", "file://")):
            return candidate

        path_obj = Path(candidate)
        if path_obj.is_absolute():
            return str(path_obj)

        suffix = path_obj.suffix.lower()
        if suffix in PREVIEW_IMAGE_EXTENSIONS:
            abs_path = (_get_active_project_path() / path_obj).resolve()
            return str(abs_path)

    return ""


def normalize_object_id(name: str, fallback_index: int) -> str:
    """将物体名转换为 object_id 友好的目录名。"""
    cleaned = re.sub(r"\s+", "_", (name or "").strip())
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "_", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = f"object_{fallback_index:02d}"
    return cleaned[:64]


def get_tool(name: str) -> Any:
    """从工具注册表中按名称获取工具，按需触发懒加载。"""
    registry = get_tool_registry()
    tools = registry.list_tools()
    if not tools:
        from ai_tools.load_tools import load_tools

        load_tools(get_ai_config())
        tools = registry.list_tools()
    return {t.name: t for t in tools}.get(name)


def get_search_tool():
    """获取物体搜索工具。"""
    # return None  # TEMP: 临时屏蔽嵌入模型，跳过检索阶段
    return get_tool("search_similar_object")


def get_store_tool():
    """获取物体入库工具。"""
    return get_tool("store_object")


def get_3d_generate_tool():
    """获取 3D 模型生成工具。优先混元3D，回退 Rodin。"""
    tool = get_tool("hunyuan_generate_3d")
    if tool is None:
        tool = get_tool("rodin_generate_3d")
    return tool


def parse_tool_result(raw_result: Any) -> Dict[str, Any]:
    """解析工具 envelope，统一返回字典结构。"""
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, str):
        return json.loads(raw_result)
    raise TypeError(f"不支持的工具返回类型: {type(raw_result)!r}")


def extract_tool_error(parsed_result: Dict[str, Any]) -> str:
    """从工具 envelope 中提取错误信息。"""
    error_code = parsed_result.get("error_code", 0)
    if not error_code:
        return ""

    status_info = str(parsed_result.get("status_info", "") or "").strip()
    if status_info and status_info.lower() != "success":
        return status_info

    try:
        parts = parsed_result["llm_content"][0]["part"]
        for part in parts:
            text = str(part.get("content_text", "") or "").strip()
            if text:
                return text
    except (KeyError, IndexError, TypeError):
        pass

    return "工具调用失败"


def parse_3d_result(raw_result: Any) -> Dict[str, Any]:
    """解析 rodin_generate_3d 返回值，提取模型文件路径与元数据。"""
    try:
        parsed = parse_tool_result(raw_result)
        error_message = extract_tool_error(parsed)
        if error_message:
            return {"error": error_message}

        metadata = parsed.get("metadata") or {}
        model_folder: str = metadata.get("model_folder", "")
        has_mesh_pending: bool = metadata.get("has_mesh_pending", False)
        folder_object_id: str = metadata.get("folder_object_id", "") or metadata.get(
            "object_id", ""
        )
        mesh_object_id: str = metadata.get("mesh_object_id", "") or metadata.get(
            "model_object_id", ""
        )

        parts = parsed["llm_content"][0]["part"]
        preview_paths: List[str] = []
        geometry_file_format = "glb"

        for part in parts:
            if part.get("content_type") == "image":
                preview_path = _resolve_preview_part_url(part)
                if preview_path:
                    preview_paths.append(preview_path)
                part_param = part.get("parameter") or {}
                fmt = part_param.get("geometry_file_format", "")
                if fmt:
                    geometry_file_format = fmt

        geometry_file_format = metadata.get(
            "geometry_file_format",
            geometry_file_format,
        )

        if model_folder:
            file_stem = mesh_object_id or folder_object_id or "base"
            model_path = f"{model_folder}/{file_stem}.{geometry_file_format}"
            parameter: Dict[str, Any] = {
                "preview_paths": preview_paths,
                "model_folder": model_folder,
                "geometry_file_format": geometry_file_format,
                "has_mesh_pending": has_mesh_pending,
                "object_id": folder_object_id,
                "folder_object_id": folder_object_id,
                "mesh_object_id": mesh_object_id,
            }
            return {
                "model_path": model_path,
                "parameter": parameter,
            }
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"error": "3D 生成结果解析失败"}


def pick_first_preview_path(*candidates: Any) -> str:
    """从若干候选图片路径集合中选择第一条可用路径。"""
    for candidate in candidates:
        if isinstance(candidate, str):
            text = candidate.strip()
            if text:
                return text
            continue

        if isinstance(candidate, list):
            for item in candidate:
                text = str(item or "").strip()
                if text:
                    return text
    return ""


def find_sibling_preview_image(model_path: str) -> str:
    """在模型同目录中查找第一张预览图（不递归）。"""
    path_text = str(model_path or "").strip()
    if not path_text:
        return ""

    lowered = path_text.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return ""

    model_file = Path(path_text)
    model_dir = model_file if model_file.is_dir() else model_file.parent
    if not model_dir.exists() or not model_dir.is_dir():
        return ""

    try:
        for entry in sorted(model_dir.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() in PREVIEW_IMAGE_EXTENSIONS:
                return str(entry)
    except OSError:
        return ""

    return ""


def build_placeholder_embedding(
    object_id: str,
    model_path: str,
    vector_dim: int,
) -> np.ndarray:
    """生成可复现的伪向量兜底，仅在嵌入模型不可用时使用。"""
    import hashlib

    seed_text = f"{object_id}|{model_path}"
    seed_bytes = hashlib.sha256(seed_text.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(vector_dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    return vec
