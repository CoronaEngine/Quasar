"""
物体识别模块 —— 工具装配层

在核心业务逻辑（execution.py）上组装 LangChain StructuredTool，
为上游工作流系统提供工具化接口。

三层架构:
    base.py (HTTP 入口) ──┐
                          ├─> execution.py (核心逻辑)
    recognition_tools.py (工具装配) ──┘
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .execution import (
    core_execute_object_store,
    core_execute_object_search,
    extract_recognition_config,
)

logger = logging.getLogger(__name__)

# 自动扫描只在当前进程中执行一次，避免重复全量扫描目录。
_AUTO_SCAN_INIT_LOCK = threading.Lock()
_AUTO_SCAN_DONE = False


# ====================================================================== #
#  工具输入模式定义
# ====================================================================== #


class StoreObjectInput(BaseModel):
    """物体入库工具的输入参数"""

    object_id: str = Field(..., description="物体唯一标识符")
    image_paths: list[str] = Field(
        ..., description="物体图片路径列表，支持多张（建议 1-6 张）"
    )
    name: Optional[str] = Field(default="", description="物体名称")
    category: Optional[str] = Field(default="", description="物体分类标签")
    description: Optional[str] = Field(default="", description="物体文字描述")


class SearchObjectInput(BaseModel):
    """物体搜索工具的输入参数"""

    query_images: Optional[list[str]] = Field(
        default=None,
        description="查询图片路径列表（可选）",
    )
    query_text: Optional[str] = Field(
        default="",
        description="查询文字描述（可选）",
    )
    top_k: Optional[int] = Field(
        default=5,
        description="返回结果数量，范围 1-20",
    )


# ====================================================================== #
#  工具包装函数
# ====================================================================== #


def _tool_store_object(
    cfg,
    object_id: str,
    image_paths: list[str],
    name: str = "",
    category: str = "",
    description: str = "",
) -> dict[str, Any]:
    """
    工具包装：物体入库

    调用核心执行函数，处理错误并返回序列化结果。
    支持图片自动去重和数量限制。
    """
    from ai_tools.response_adapter import build_success_result, build_error_result

    result = core_execute_object_store(
        cfg=cfg,
        object_id=object_id,
        image_paths=image_paths,
        name=name,
        category=category,
        description=description,
        dedup=True,  # 自动去重
        max_images=6,  # 自动限制到6张
    )

    # 错误处理
    if "error" in result:
        return build_error_result(error_message=result["error"]).to_dict(
            interface_type="object_recognition"
        )

    # 成功返回：返回标准化格式，供工作流直接使用
    parts = result.get("parts", [])
    response = build_success_result(parts=parts).to_dict(
        interface_type="object_recognition"
    )

    response["register_status"] = result.get("register_status", "inserted")
    response["rowid"] = result.get("rowid")
    response["object_id"] = object_id

    return response


def _tool_search_object(
    cfg,
    query_images: Optional[list[str]] = None,
    query_text: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    工具包装：物体搜索

    调用核心执行函数，处理错误并返回序列化结果。
    内化距离阈值判断，返回"hit"字段供工作流使用。
    """
    from ai_tools.response_adapter import build_success_result, build_error_result

    result = core_execute_object_search(
        cfg=cfg,
        query_images=query_images,
        query_text=query_text,
        top_k=top_k,
        distance_threshold=0.3,  # 内化threshold，可配置化
    )

    # 错误处理
    if "error" in result:
        return build_error_result(error_message=result["error"]).to_dict(
            interface_type="object_recognition"
        )

    # 成功返回：返回标准化格式，供工作流直接使用
    parts = result.get("parts", [])
    response = build_success_result(parts=parts).to_dict(
        interface_type="object_recognition"
    )

    response["hit"] = result.get("hit", False)
    response["best_match"] = result.get("best_match")
    response["all_matches"] = result.get("all_matches", [])

    return response


# ====================================================================== #
#  工具加载
# ====================================================================== #


def load_recognition_tools(config) -> list[StructuredTool]:
    """
    加载物体识别相关工具（StructuredTool 列表）。

    参数:
        config: AI 全局配置对象

    返回:
        StructuredTool 列表（store_object, search_similar_object）

    说明:
        根据配置决定是否启用模块。若启用，则初始化嵌入模型和向量数据库，
        并返回可用的工具。若禁用，则返回空列表。
    """
    recognition_cfg = extract_recognition_config(config)

    if not recognition_cfg.enable:
        logger.info("物体识别模块未启用，返回空工具列表")
        return []

    logger.info("初始化物体识别工具")

    # 尝试在工具加载阶段执行一次自动扫描入库。
    # 失败时仅记录日志，不阻塞工具注册和后续使用。
    global _AUTO_SCAN_DONE
    if not _AUTO_SCAN_DONE:
        with _AUTO_SCAN_INIT_LOCK:
            if not _AUTO_SCAN_DONE:
                try:
                    from .auto_scan import scan_and_register
                    from .client_embedding import build_provider
                    from .vector_db import get_vector_db

                    embedding_provider = build_provider(recognition_cfg)
                    vector_db = get_vector_db(
                        db_path=recognition_cfg.vector_db.db_path,
                        vector_dim=recognition_cfg.vector_db.vector_dim,
                    )
                    scan_stats = scan_and_register(
                        recognition_cfg=recognition_cfg,
                        vector_db=vector_db,
                        embedding_client=embedding_provider,
                    )
                    logger.info(f"物体目录自动扫描完成: {scan_stats}")
                except Exception as e:
                    logger.warning(f"物体目录自动扫描失败（不影响工具加载）: {e}")
                finally:
                    _AUTO_SCAN_DONE = True

    try:
        # 工具 1: 物体入库
        tool_store_object = StructuredTool.from_function(
            func=lambda object_id, image_paths, name="", category="", description="": _tool_store_object(
                cfg=config,
                object_id=object_id,
                image_paths=image_paths,
                name=name,
                category=category,
                description=description,
            ),
            name="store_object",
            description=(
                "将物体及其图片存入向量数据库，用于后续相似物体查询。"
                "支持使用多张物体图片或文字描述。"
            ),
            args_schema=StoreObjectInput,
        )

        # 工具 2: 物体搜索
        tool_search_object = StructuredTool.from_function(
            func=lambda query_images=None, query_text="", top_k=5: _tool_search_object(
                cfg=config,
                query_images=query_images,
                query_text=query_text,
                top_k=top_k,
            ),
            name="search_similar_object",
            description=(
                "根据提供的查询图片或文字描述，在向量数据库中搜索相似物体。"
                "返回最相似的 top_k 个物体及其相似度分数。"
            ),
            args_schema=SearchObjectInput,
        )

        tools = [tool_store_object, tool_search_object]
        logger.info(f"物体识别工具加载成功，计 {len(tools)} 个工具")
        return tools

    except Exception as e:
        logger.error(f"加载物体识别工具失败: {e}")
        return []


__all__ = [
    "load_recognition_tools",
    "StoreObjectInput",
    "SearchObjectInput",
]
