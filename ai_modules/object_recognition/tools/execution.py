"""
物体识别模块 —— 核心业务逻辑

包含物体入库、搜索的纯业务实现，配置提取统一来源。
该层不依赖 HTTP 协议，可被工具化适配层（recognition_tools.py）或 HTTP 入口（base.py）共同调用。

依赖:
    - Dashscope SDK for embedding
    - sqlite-vec for vector storage
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ....ai_config.ai_config import AIConfig
from ....ai_tools.response_adapter import (
    build_part,
)
from ..configs.dataclasses import (
    EmbeddingModelConfig,
    VectorDBConfig,
    RecognitionConfig,
)

logger = logging.getLogger(__name__)


# ====================================================================== #
#  配置提取（统一单一来源）
# ====================================================================== #


def extract_recognition_config(config: AIConfig) -> RecognitionConfig:
    """
    从 AIConfig 中提取并构建 RecognitionConfig。

    优先使用 AIConfig 中的配置；若不存在则使用默认值。
    这是唯一的配置转换来源，避免与其他模块的实现漂移。
    """
    try:
        raw = getattr(config, "object_recognition", None)
        if raw is None:
            logger.debug("AIConfig 中无 object_recognition 配置，使用默认值")
            return RecognitionConfig()

        if isinstance(raw, dict):
            defaults = RecognitionConfig()
            # 从字典构建配置
            embedding_raw = raw.get("embedding", {})
            vector_db_raw = raw.get("vector_db", {})

            embedding_cfg = (
                EmbeddingModelConfig(**embedding_raw)
                if embedding_raw
                else EmbeddingModelConfig()
            )
            vector_db_cfg = (
                VectorDBConfig(**vector_db_raw) if vector_db_raw else VectorDBConfig()
            )

            return RecognitionConfig(
                enable=raw.get("enable", defaults.enable),
                provider=raw.get("provider", defaults.provider),
                embedding=embedding_cfg,
                vector_db=vector_db_cfg,
                standard_image_count=raw.get("standard_image_count", defaults.standard_image_count),
                storage_instruction=raw.get(
                    "storage_instruction",
                    defaults.storage_instruction,
                ),
                query_instruction=raw.get(
                    "query_instruction",
                    defaults.query_instruction,
                ),
                dashscope_api_key=raw.get("dashscope_api_key", defaults.dashscope_api_key),
                dashscope_model=raw.get(
                    "dashscope_model",
                    defaults.dashscope_model,
                ),
                auto_scan_dir=raw.get("auto_scan_dir", defaults.auto_scan_dir),
                auto_scan_embed=raw.get("auto_scan_embed", defaults.auto_scan_embed),
                auto_scan_max_images=raw.get("auto_scan_max_images", defaults.auto_scan_max_images),
            )

        if isinstance(raw, RecognitionConfig):
            return raw

        logger.warning(f"意外的配置类型: {type(raw)}，使用默认值")
        return RecognitionConfig()

    except Exception as e:
        logger.error(f"解析物体识别配置失败: {e}，使用默认值")
        return RecognitionConfig()


def _resolve_embedding_api_key(config: AIConfig, recognition_cfg: RecognitionConfig) -> str:
    """优先从 providers 解析嵌入 API Key，兼容旧字段回退。"""
    provider_name = (getattr(recognition_cfg, "provider", "") or "").strip()
    if provider_name:
        try:
            providers = getattr(config, "providers", {}) or {}
            provider = providers.get(provider_name)
            if provider and getattr(provider, "api_key", None):
                return str(provider.api_key)
        except Exception as e:
            logger.warning("从 providers 解析 object_recognition api_key 失败: %s", e)

    return (getattr(recognition_cfg, "dashscope_api_key", "") or "").strip()


# ====================================================================== #
#  核心执行逻辑（不依赖 HTTP 协议）
# ====================================================================== #


def core_execute_object_store(
    cfg,
    object_id: str,
    image_paths: List[str],
    name: str = "",
    category: str = "",
    description: str = "",
    dedup: bool = True,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """
    物体入库核心逻辑（由 HTTP 入口与 StructuredTool 共同调用）。

    参数:
        cfg: AI 全局配置对象
        object_id: 物体唯一标识
        image_paths: 图片路径列表
        name: 物体名称
        category: 物体分类
        description: 文字描述
        dedup: 是否去重图片路径（默认True）
        max_images: 最大图片数量限制（默认None=不限）

    返回:
        {
            "error": "错误信息" (若失败)
            "parts": [部分列表] (若成功)
            "register_status": "inserted" | "updated" (若成功)
            "rowid": 数据库行号 (若成功)
        }
    """
    recognition_cfg = extract_recognition_config(cfg)

    if not recognition_cfg.enable:
        return {"error": "物体识别模块未启用"}

    try:
        # 预处理图片路径：去重 + 限制数量
        if dedup:
            unique_paths = list(dict.fromkeys(image_paths))
        else:
            unique_paths = image_paths

        if max_images is None:
            max_images = recognition_cfg.standard_image_count

        final_paths = unique_paths[:max_images]

        logger.info(
            f"开始物体入库（核心）: object_id={object_id}, "
            f"images={len(image_paths)} → {len(final_paths)}, name={name}"
        )

        # 图片数量验证（允许少于 6 张，自动降级）
        if len(final_paths) > recognition_cfg.standard_image_count:
            logger.warning(
                f"图片数量 ({len(final_paths)}) 超过标准六面图数量 "
                f"({recognition_cfg.standard_image_count})，将使用全部图片"
            )

        if not image_paths and not description:
            return {"error": "入库失败: 至少需要提供图片或文字描述"}

        # 延迟导入依赖
        from .client_embedding import build_provider
        from .vector_db import get_vector_db

        # 初始化嵌入提供者和向量数据库
        recognition_cfg.dashscope_api_key = _resolve_embedding_api_key(cfg, recognition_cfg)
        embedding_provider = build_provider(recognition_cfg)
        vector_db = get_vector_db(
            db_path=recognition_cfg.vector_db.db_path,
            vector_dim=recognition_cfg.vector_db.vector_dim,
        )

        # 生成存储侧嵌入向量（多图 + 文本融合为单一向量）
        embedding = embedding_provider.embed_for_storage(
            image_paths=final_paths,
            text=description,
            instruction=recognition_cfg.storage_instruction,
        )

        # 写入向量数据库（存在则更新，不存在则插入）
        existing = vector_db.get_object(object_id)
        register_status = "inserted"
        rowid = None
        if existing is None:
            rowid = vector_db.insert_object(
                object_id=object_id,
                embedding=embedding,
                name=name,
                category=category,
                image_paths=image_paths,
                description=description,
            )
        else:
            updated = vector_db.update_object(
                object_id=object_id,
                embedding=embedding,
                name=name,
                category=category,
                image_paths=image_paths,
                description=description,
            )
            if not updated:
                return {"error": f"入库失败: 物体 '{object_id}' 更新失败"}
            register_status = "updated"

        # 构建成功响应（返回结构化对象，由调用方序列化）
        result_text = (
            f"物体入库成功\n"
            f"- object_id: {object_id}\n"
            f"- 名称: {name or '(未命名)'}\n"
            f"- 分类: {category or '(未分类)'}\n"
            f"- 图片数: {len(image_paths)}\n"
            f"- 向量维度: {recognition_cfg.embedding.output_dim}\n"
            f"- 写入方式: {'插入' if register_status == 'inserted' else '更新'}\n"
            f"- 数据库行号: {rowid if rowid is not None else '(更新无新行号)'}"
        )

        part = build_part(
            content_type="text",
            content_text=result_text,
            parameter={
                "object_id": object_id,
                "rowid": rowid,
                "register_status": register_status,
                "image_count": len(final_paths),
                "vector_dim": recognition_cfg.embedding.output_dim,
            },
        )

        # 返回标准化结果（供工具和HTTP入口使用）
        return {
            "parts": [part],
            "register_status": register_status,
            "rowid": rowid,
        }

    except ValueError as e:
        logger.error(f"物体入库参数错误: {e}")
        return {"error": f"入库失败: {e}"}
    except Exception as e:
        logger.error(f"物体入库异常: {e}")
        return {"error": f"入库异常: {e}"}


def core_execute_object_search(
    cfg,
    query_images: Optional[List[str]] = None,
    query_text: str = "",
    top_k: int = 5,
    distance_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    物体搜索核心逻辑（由 HTTP 入口与 StructuredTool 共同调用）。

    参数:
        cfg: AI 全局配置对象
        query_images: 查询图片路径列表
        query_text: 查询文字描述
        top_k: 返回结果数量
        distance_threshold: 距离阈值，小于此值视为命中（默认0.3）

    返回:
        {
            "error": "错误信息" (若失败)
            "parts": [部分列表] (若成功)
            "hit": True|False (是否命中，基于distance_threshold)
            "best_match": {...} (最佳匹配项，若命中)
            "all_matches": [...] (所有匹配项)
        }
    """
    recognition_cfg = extract_recognition_config(cfg)

    if not recognition_cfg.enable:
        return {"error": "物体识别模块未启用"}

    try:
        images = query_images or []
        logger.info(
            f"开始物体搜索（核心）: images={len(images)}, "
            f"text_len={len(query_text)}, top_k={top_k}, threshold={distance_threshold}"
        )

        if not images and not query_text:
            return {"error": "搜索失败: 至少需要提供查询图片或文字描述"}

        # 延迟导入依赖
        from .client_embedding import build_provider
        from .vector_db import get_vector_db

        # 初始化嵌入提供者和向量数据库
        embedding_provider = build_provider(recognition_cfg)
        vector_db = get_vector_db(
            db_path=recognition_cfg.vector_db.db_path,
            vector_dim=recognition_cfg.vector_db.vector_dim,
        )

        # 生成查询侧嵌入向量
        query_embedding = embedding_provider.embed_for_query(
            image_paths=images if images else None,
            text=query_text if query_text else None,
            instruction=recognition_cfg.query_instruction,
        )

        # 向量检索
        results = vector_db.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not results:
            result_text = "未找到相似物体"
            part = build_part(
                content_type="text",
                content_text=result_text,
                parameter={"matches": [], "total": 0},
            )
            return {
                "parts": [part],
                "hit": False,
                "best_match": None,
                "all_matches": [],
            }

        # 判断是否命中（基于threshold）
        best = results[0]
        best_distance = best.get("distance", 999)
        hit = best_distance < distance_threshold

        # 构建结果文本
        lines = [f"找到 {len(results)} 个相似物体：\n"]
        match_list = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"  {i}. [{r['object_id']}] {r['name'] or '(未命名)'}"
                f"  (距离: {r['distance']:.6f})"
                f"  分类: {r['category'] or '(无)'}"
            )
            match_list.append(
                {
                    "rank": i,
                    "object_id": r["object_id"],
                    "name": r["name"],
                    "category": r["category"],
                    "distance": r["distance"],
                    "description": r["description"],
                }
            )

        result_text = "\n".join(lines)
        part = build_part(
            content_type="text",
            content_text=result_text,
            parameter={
                "matches": match_list,
                "total": len(results),
                "query_images_count": len(images),
                "query_text": query_text,
                "hit": hit,
                "best_distance": best_distance,
                "threshold": distance_threshold,
            },
        )

        # 返回标准化结果（供工具和HTTP入口使用）
        return {
            "parts": [part],
            "hit": hit,
            "best_match": best if hit else None,
            "all_matches": results,
        }

    except Exception as e:
        logger.error(f"物体搜索异常: {e}")
        return {"error": f"搜索异常: {e}"}


__all__ = [
    "extract_recognition_config",
    "core_execute_object_store",
    "core_execute_object_search",
]
