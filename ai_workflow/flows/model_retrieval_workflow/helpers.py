from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import numpy as np

from ai_config.ai_config import get_ai_config
from ai_tools.registry import get_tool_registry
from config.app_config import get_app_config

logger = logging.getLogger(__name__)


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
    return get_tool("search_similar_object")


def get_3d_generate_tool():
    """获取 3D 模型生成工具。"""
    return get_tool("rodin_generate_3d")


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


def parse_search_result(raw_result: Any) -> Dict[str, Any]:
    """解析 search_similar_object 返回值。"""
    try:
        parsed = parse_tool_result(raw_result)
        error_message = extract_tool_error(parsed)
        if error_message:
            return {"matches": [], "error": error_message}

        parts = parsed["llm_content"][0]["part"]
        for part in parts:
            matches = part.get("parameter", {}).get("matches", [])
            if isinstance(matches, list):
                return {"matches": matches, "error": ""}
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"matches": [], "error": "搜索结果解析失败"}


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
        meta_object_id: str = metadata.get("object_id", "")

        parts = parsed["llm_content"][0]["part"]
        preview_paths: List[str] = []
        geometry_file_format = "glb"

        for part in parts:
            if part.get("content_type") == "image":
                preview_path = part.get("content_text") or part.get("content_url") or ""
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
            model_path = f"{model_folder}/base.{geometry_file_format}"
            parameter: Dict[str, Any] = {
                "preview_paths": preview_paths,
                "model_folder": model_folder,
                "geometry_file_format": geometry_file_format,
                "has_mesh_pending": has_mesh_pending,
                "object_id": meta_object_id,
            }
            return {
                "model_path": model_path,
                "parameter": parameter,
            }
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"error": "3D 生成结果解析失败"}


def get_recognition_db_config() -> Dict[str, Any]:
    """读取 object_recognition 向量库配置。"""
    cfg = get_ai_config()
    raw = getattr(cfg, "object_recognition", None)

    db_path = str(get_app_config().paths.object_recognition_db)
    vector_dim = 1024

    if isinstance(raw, dict):
        vector_cfg = raw.get("vector_db", {}) or {}
        db_path = str(vector_cfg.get("db_path", db_path))
        vector_dim = int(vector_cfg.get("vector_dim", vector_dim))
    elif raw is not None:
        vector_cfg = getattr(raw, "vector_db", None)
        if vector_cfg is not None:
            db_path = str(getattr(vector_cfg, "db_path", db_path))
            vector_dim = int(getattr(vector_cfg, "vector_dim", vector_dim))

    return {
        "db_path": db_path,
        "vector_dim": vector_dim,
    }


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


def get_embedding_client():
    """获取 Qwen3-VL-Embedding 客户端单例，按需从配置初始化。"""
    from ai_modules.object_recognition.configs.dataclasses import (
        EmbeddingModelConfig,
        RecognitionConfig,
    )
    from ai_modules.object_recognition.tools.client_embedding import (
        get_embedding_client,
    )

    cfg = get_ai_config()
    raw = getattr(cfg, "object_recognition", None)

    if isinstance(raw, dict):
        embedding_raw = raw.get("embedding", {}) or {}
        embedding_cfg = (
            EmbeddingModelConfig(**embedding_raw)
            if embedding_raw
            else EmbeddingModelConfig()
        )
    elif isinstance(raw, RecognitionConfig):
        embedding_cfg = raw.embedding
    else:
        embedding_cfg = EmbeddingModelConfig()

    return get_embedding_client(embedding_cfg)
