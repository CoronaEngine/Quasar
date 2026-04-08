"""
物体识别模块 —— LangChain 工具加载器

将 embed_and_store_object / search_similar_object 封装为 LangChain StructuredTool，
供 ToolRegistry 统一管理和调度。

依赖:
    pip install langchain-core pydantic numpy sqlite-vec
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ai_config.ai_config import AIConfig
from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)
from ..configs.dataclasses import (
    EmbeddingModelConfig,
    VectorDBConfig,
    RecognitionConfig,
)
from ..configs.prompts import (
    STORE_OBJECT_PROMPTS,
    SEARCH_OBJECT_PROMPTS,
)

logger = logging.getLogger(__name__)

# ====================================================================== #
#  工具加载函数
# ====================================================================== #


def load_recognition_tools(config: AIConfig) -> List[Any]:
    """
    加载物体识别相关工具。

    根据配置决定是否启用模块，初始化嵌入模型客户端和向量数据库，
    并返回封装好的 StructuredTool 列表。

    参数:
        config: AI 全局配置

    返回:
        StructuredTool 列表（包含入库工具和搜索工具）
    """
    # 从全局配置中读取本模块配置
    recognition_cfg = _extract_recognition_config(config)

    if not recognition_cfg.enable:
        logger.info("物体识别模块未启用，跳过工具加载")
        return []

    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise RuntimeError(
            "物体识别模块依赖缺失，请安装: pip install langchain-core pydantic"
        ) from e

    class StoreObjectInput(BaseModel):
        object_id: str = Field(
            ...,
            description=STORE_OBJECT_PROMPTS.fields["object_id"],
        )
        image_paths: List[str] = Field(
            ...,
            description=STORE_OBJECT_PROMPTS.fields["image_paths"],
        )
        name: str = Field(
            default="",
            description=STORE_OBJECT_PROMPTS.fields["name"],
        )
        category: str = Field(
            default="",
            description=STORE_OBJECT_PROMPTS.fields["category"],
        )
        description: str = Field(
            default="",
            description=STORE_OBJECT_PROMPTS.fields["description"],
        )

    class SearchObjectInput(BaseModel):
        query_images: List[str] = Field(
            default_factory=list,
            description=SEARCH_OBJECT_PROMPTS.fields["query_images"],
        )
        query_text: str = Field(
            default="",
            description=SEARCH_OBJECT_PROMPTS.fields["query_text"],
        )
        top_k: int = Field(
            default=5,
            description=SEARCH_OBJECT_PROMPTS.fields["top_k"],
        )

    # 延迟导入，避免在模块未启用时加载重型依赖
    from .client_embedding import (
        get_embedding_client,
    )
    from .vector_db import VectorDB

    # 初始化嵌入客户端和向量数据库
    embedding_client = get_embedding_client(recognition_cfg.embedding)
    vector_db = VectorDB(
        db_path=recognition_cfg.vector_db.db_path,
        vector_dim=recognition_cfg.vector_db.vector_dim,
    )

    logger.info(
        f"物体识别工具初始化完成: "
        f"model={recognition_cfg.embedding.model_size}, "
        f"dim={recognition_cfg.embedding.output_dim}, "
        f"db={recognition_cfg.vector_db.db_path}"
    )

    # ── 目录自动扫描（模块加载时执行一次） ──────────────────────────
    if recognition_cfg.auto_scan_dir:
        try:
            from .auto_scan import (
                scan_and_register,
            )

            scan_and_register(
                recognition_cfg=recognition_cfg,
                vector_db=vector_db,
                embedding_client=embedding_client,
            )
        except Exception as e:
            logger.error(f"目录自动扫描异常（不影响工具加载）: {e}")

    # ── 物体入库工具 ────────────────────────────────────────────────

    def _store_object(
        object_id: str,
        image_paths: List[str],
        name: str = "",
        category: str = "",
        description: str = "",
        **kwargs,
    ) -> str:
        """物体入库：将六面图 + 描述融合为嵌入向量并存储"""
        try:
            logger.info(
                f"开始物体入库: object_id={object_id}, "
                f"images={len(image_paths)}, name={name}"
            )

            # 图片数量验证（允许少于 6 张，自动降级）
            if len(image_paths) > recognition_cfg.standard_image_count:
                logger.warning(
                    f"图片数量 ({len(image_paths)}) 超过标准六面图数量 "
                    f"({recognition_cfg.standard_image_count})，将使用全部图片"
                )

            if not image_paths and not description:
                return build_error_result(
                    error_message="入库失败: 至少需要提供图片或文字描述"
                ).to_envelope(interface_type="object_recognition")

            # 生成存储侧嵌入向量（多图 + 文本融合为单一向量）
            embedding = embedding_client.embed_for_storage(
                image_paths=image_paths,
                text=description,
            )

            # 写入向量数据库
            rowid = vector_db.insert_object(
                object_id=object_id,
                embedding=embedding,
                name=name,
                category=category,
                image_paths=image_paths,
                description=description,
            )

            # 构建成功响应
            result_text = (
                f"物体入库成功\n"
                f"- object_id: {object_id}\n"
                f"- 名称: {name or '(未命名)'}\n"
                f"- 分类: {category or '(未分类)'}\n"
                f"- 图片数: {len(image_paths)}\n"
                f"- 向量维度: {recognition_cfg.embedding.output_dim}\n"
                f"- 数据库行号: {rowid}"
            )

            part = build_part(
                content_type="text",
                content_text=result_text,
                parameter={
                    "object_id": object_id,
                    "rowid": rowid,
                    "image_count": len(image_paths),
                    "vector_dim": recognition_cfg.embedding.output_dim,
                },
            )
            return build_success_result(parts=[part]).to_envelope(
                interface_type="object_recognition"
            )

        except ValueError as e:
            logger.error(f"物体入库参数错误: {e}")
            return build_error_result(error_message=f"入库失败: {e}").to_envelope(
                interface_type="object_recognition"
            )
        except Exception as e:
            logger.error(f"物体入库异常: {e}")
            return build_error_result(error_message=f"入库异常: {e}").to_envelope(
                interface_type="object_recognition"
            )

    # ── 物体搜索工具 ────────────────────────────────────────────────

    def _search_object(
        query_images: Optional[List[str]] = None,
        query_text: str = "",
        top_k: int = 5,
        **kwargs,
    ) -> str:
        """物体搜索：根据图片/文字查询最相似的物体"""
        try:
            images = query_images or []
            logger.info(
                f"开始物体搜索: images={len(images)}, "
                f"text_len={len(query_text)}, top_k={top_k}"
            )

            if not images and not query_text:
                return build_error_result(
                    error_message="搜索失败: 至少需要提供查询图片或文字描述"
                ).to_envelope(interface_type="object_recognition")

            # 生成查询侧嵌入向量
            query_embedding = embedding_client.embed_for_query(
                image_paths=images if images else None,
                text=query_text if query_text else None,
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
                return build_success_result(parts=[part]).to_envelope(
                    interface_type="object_recognition"
                )

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
                },
            )
            return build_success_result(parts=[part]).to_envelope(
                interface_type="object_recognition"
            )

        except Exception as e:
            logger.error(f"物体搜索异常: {e}")
            return build_error_result(error_message=f"搜索异常: {e}").to_envelope(
                interface_type="object_recognition"
            )

    # ── 构建 StructuredTool 列表 ────────────────────────────────────

    store_tool = StructuredTool(
        name="store_object",
        description=STORE_OBJECT_PROMPTS.tool_description,
        args_schema=StoreObjectInput,
        func=_store_object,
    )

    search_tool = StructuredTool(
        name="search_similar_object",
        description=SEARCH_OBJECT_PROMPTS.tool_description,
        args_schema=SearchObjectInput,
        func=_search_object,
    )

    logger.info("物体识别工具加载完成: store_object, search_similar_object")
    return [store_tool, search_tool]


# ====================================================================== #
#  辅助函数
# ====================================================================== #


def _extract_recognition_config(config: AIConfig) -> RecognitionConfig:
    """
    从 AIConfig 中提取并构建 RecognitionConfig。

    优先使用 AIConfig 中的配置；若不存在则使用默认值。
    """
    try:
        raw = getattr(config, "object_recognition", None)
        if raw is None:
            logger.debug("AIConfig 中无 object_recognition 配置，使用默认值")
            return RecognitionConfig()

        if isinstance(raw, dict):
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
                enable=raw.get("enable", False),
                embedding=embedding_cfg,
                vector_db=vector_db_cfg,
                standard_image_count=raw.get("standard_image_count", 6),
                storage_instruction=raw.get(
                    "storage_instruction",
                    "Represent this document for retrieval:",
                ),
                query_instruction=raw.get(
                    "query_instruction",
                    "Represent the query for retrieving relevant documents:",
                ),
                auto_scan_dir=raw.get("auto_scan_dir", ""),
                auto_scan_embed=raw.get("auto_scan_embed", False),
                auto_scan_max_images=raw.get("auto_scan_max_images", 6),
            )

        if isinstance(raw, RecognitionConfig):
            return raw

        logger.warning(f"意外的配置类型: {type(raw)}，使用默认值")
        return RecognitionConfig()

    except Exception as e:
        logger.error(f"解析物体识别配置失败: {e}，使用默认值")
        return RecognitionConfig()


__all__ = [
    "load_recognition_tools",
]
