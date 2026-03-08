"""
物体识别模块 —— 目录自动扫描入库

在模块加载时扫描指定目录，检查其一级子文件夹是否已在向量数据库中登记。
对于未登记的子文件夹：
  - 若 auto_scan_embed 开关开启：自动读取子文件夹内图片（最多 N 张），
    生成嵌入向量并入库。
  - 若开关关闭：仅输出 WARNING 级别日志，不做实际入库操作。

子文件夹名同时作为 object_id 和 name。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_modules.object_recognition.configs.dataclasses import RecognitionConfig
    from ai_modules.object_recognition.tools.client_embedding import (
        Qwen3VLEmbeddingClient,
    )
    from ai_modules.object_recognition.tools.vector_db import VectorDB

logger = logging.getLogger(__name__)

# 支持的图片扩展名（与 qwen3_vl_embedding.is_image_path 保持一致）
_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif",
    ".bmp", ".webp", ".tiff", ".svg",
}


# ====================================================================== #
#  公共 API
# ====================================================================== #


def scan_and_register(
    recognition_cfg: "RecognitionConfig",
    vector_db: "VectorDB",
    embedding_client: "Qwen3VLEmbeddingClient",
) -> Dict[str, Any]:
    """
    扫描 auto_scan_dir 下的一级子文件夹，将未登记的物体入库。

    参数:
        recognition_cfg:  物体识别整体配置（含 auto_scan_* 字段）
        vector_db:        已初始化的向量数据库实例
        embedding_client: 已初始化的嵌入模型客户端

    返回:
        扫描统计摘要:
        {
            "scanned":    int,   # 扫描到的子文件夹总数
            "already_registered": int,  # 已在数据库中的数量
            "registered": int,   # 本次成功入库数量
            "warned":     int,   # 因开关关闭仅输出警告的数量
            "skipped":    int,   # 因无图片等原因跳过的数量
            "errors":     list,  # 入库失败的详情 [{object_id, error}, ...]
        }
    """
    scan_dir = recognition_cfg.auto_scan_dir
    auto_embed = recognition_cfg.auto_scan_embed
    max_images = recognition_cfg.auto_scan_max_images

    stats: Dict[str, Any] = {
        "scanned": 0,
        "already_registered": 0,
        "registered": 0,
        "warned": 0,
        "skipped": 0,
        "errors": [],
    }

    # ── 验证扫描目录 ──
    if not scan_dir:
        logger.debug("auto_scan_dir 未配置，跳过目录扫描")
        return stats

    scan_dir = os.path.abspath(scan_dir)
    if not os.path.isdir(scan_dir):
        logger.warning(f"auto_scan_dir 不存在或不是目录: {scan_dir}")
        return stats

    # ── 枚举一级子文件夹 ──
    try:
        entries = sorted(os.listdir(scan_dir))
    except OSError as e:
        logger.error(f"无法读取扫描目录 {scan_dir}: {e}")
        return stats

    subdirs = [
        name for name in entries
        if os.path.isdir(os.path.join(scan_dir, name))
        and not name.startswith(".")  # 忽略隐藏文件夹
    ]

    stats["scanned"] = len(subdirs)
    if not subdirs:
        logger.info(f"扫描目录 {scan_dir} 下没有子文件夹")
        return stats

    logger.info(
        f"开始扫描目录: {scan_dir}  "
        f"(子文件夹: {len(subdirs)}, 自动入库: {auto_embed})"
    )

    # ── 逐个检查并处理 ──
    for folder_name in subdirs:
        object_id = folder_name
        folder_path = os.path.join(scan_dir, folder_name)

        # 检查是否已在数据库中登记
        existing = vector_db.get_object(object_id)
        if existing is not None:
            logger.debug(f"  已登记，跳过: {object_id}")
            stats["already_registered"] += 1
            continue

        # 收集文件夹内的图片
        image_paths = _collect_images(folder_path, max_images)

        if not image_paths:
            logger.warning(
                f"  子文件夹 '{object_id}' 内无可用图片，跳过"
            )
            stats["skipped"] += 1
            continue

        # ── 开关判断 ──
        if not auto_embed:
            logger.warning(
                f"  未登记物体: '{object_id}'  "
                f"({len(image_paths)} 张图片)  "
                f"[auto_scan_embed=False, 仅警告]"
            )
            stats["warned"] += 1
            continue

        # ── 执行嵌入 + 入库 ──
        try:
            logger.info(
                f"  自动入库: '{object_id}'  "
                f"({len(image_paths)} 张图片)"
            )
            embedding = embedding_client.embed_for_storage(
                image_paths=image_paths,
                text="",
            )
            vector_db.insert_object(
                object_id=object_id,
                embedding=embedding,
                name=object_id,
                category="",
                image_paths=image_paths,
                description="",
            )
            stats["registered"] += 1
            logger.info(f"  入库成功: '{object_id}'")

        except Exception as e:
            logger.error(f"  入库失败: '{object_id}': {e}")
            stats["errors"].append({"object_id": object_id, "error": str(e)})

    # ── 汇总日志 ──
    logger.info(
        f"目录扫描完成: "
        f"扫描 {stats['scanned']}, "
        f"已登记 {stats['already_registered']}, "
        f"新入库 {stats['registered']}, "
        f"仅警告 {stats['warned']}, "
        f"跳过 {stats['skipped']}, "
        f"失败 {len(stats['errors'])}"
    )
    return stats


# ====================================================================== #
#  内部辅助
# ====================================================================== #


def _collect_images(folder_path: str, max_count: int) -> List[str]:
    """
    收集文件夹内的图片文件路径。

    按文件名排序，最多返回 max_count 张。
    仅扫描文件夹根层级，不递归子目录。

    参数:
        folder_path: 文件夹绝对路径
        max_count:   最多返回的图片数量

    返回:
        图片绝对路径列表
    """
    try:
        files = sorted(os.listdir(folder_path))
    except OSError:
        return []

    images: List[str] = []
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        _, ext = os.path.splitext(fname.lower())
        if ext in _IMAGE_EXTENSIONS:
            images.append(fpath)
            if len(images) >= max_count:
                break

    return images


__all__ = [
    "scan_and_register",
]
