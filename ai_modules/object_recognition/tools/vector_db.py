"""
物体识别模块 —— sqlite-vec 向量数据库操作

使用 sqlite-vec 扩展实现纯本地、单 .db 文件的向量存储与检索。
所有向量在存入前会被归一化为单位向量，以便使用余弦相似度检索。

数据表结构:
    objects —— vec0 虚拟表
        id              INTEGER PRIMARY KEY
        object_id       TEXT UNIQUE          -- 物体唯一业务标识
        embedding       FLOAT[dim]           -- 归一化后的嵌入向量
        name            TEXT                 -- 物体名称
        category        TEXT                 -- 分类标签
        image_paths     TEXT                 -- 原始图片路径 JSON 列表
        description     TEXT                 -- 文字描述
        created_at      DATETIME             -- 创建时间

依赖:
    pip install sqlite-vec
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 全局数据库连接锁（sqlite 在多线程下需要序列化写入）
_DB_LOCK = threading.Lock()


class VectorDB:
    """基于 sqlite-vec 的本地向量数据库管理器"""

    def __init__(self, db_path: str, vector_dim: int = 1024) -> None:
        """
        初始化向量数据库。

        参数:
            db_path:    数据库文件路径（单 .db 文件）
            vector_dim: 向量维度，须与嵌入模型 output_dim 一致
        """
        self.db_path = os.path.abspath(db_path)
        self.vector_dim = vector_dim
        self._local = threading.local()

        # 确保目录存在
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # 首次连接时建表
        self._init_tables()
        logger.info(f"向量数据库已初始化: {self.db_path} (dim={self.vector_dim})")

    # ------------------------------------------------------------------ #
    #  连接管理
    # ------------------------------------------------------------------ #

    def _get_connection(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接（线程本地单例）"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            # 加载 sqlite-vec 扩展
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception as e:
                logger.error(f"加载 sqlite-vec 扩展失败: {e}")
                raise RuntimeError(
                    "无法加载 sqlite-vec 扩展，请确认已安装: pip install sqlite-vec"
                ) from e
            conn.enable_load_extension(False)
            self._local.conn = conn
        return conn

    def _init_tables(self) -> None:
        """初始化数据库表结构"""
        conn = self._get_connection()

        # 创建元数据表（普通表，存储物体的非向量字段）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS object_metadata (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id   TEXT UNIQUE NOT NULL,
                name        TEXT DEFAULT '',
                category    TEXT DEFAULT '',
                image_paths TEXT DEFAULT '[]',
                description TEXT DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建 vec0 虚拟表（用于向量检索）
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS object_vectors
            USING vec0(
                object_rowid INTEGER PRIMARY KEY,
                embedding FLOAT[{self.vector_dim}]
            )
        """)

        conn.commit()
        logger.debug("数据库表结构初始化完成")

    # ------------------------------------------------------------------ #
    #  写入操作
    # ------------------------------------------------------------------ #

    def insert_object(
        self,
        object_id: str,
        embedding: np.ndarray,
        name: str = "",
        category: str = "",
        image_paths: Optional[List[str]] = None,
        description: str = "",
    ) -> int:
        """
        插入一个物体的嵌入向量和元数据。

        参数:
            object_id:   物体唯一标识
            embedding:   归一化后的嵌入向量 (numpy array, shape=[dim])
            name:        物体名称
            category:    分类标签
            image_paths: 原始图片路径列表
            description: 文字描述

        返回:
            插入记录的 rowid

        异常:
            ValueError: object_id 已存在时抛出
        """
        if embedding.shape != (self.vector_dim,):
            raise ValueError(
                f"向量维度不匹配: 期望 ({self.vector_dim},)，实际 {embedding.shape}"
            )

        # 序列化图片路径为 JSON 字符串
        paths_json = json.dumps(image_paths or [], ensure_ascii=False)
        # 将向量转为字符串格式 "[0.1, 0.2, ...]"
        vec_str = _numpy_to_vec_string(embedding)

        with _DB_LOCK:
            conn = self._get_connection()
            try:
                # 先插入元数据
                cursor = conn.execute(
                    """
                    INSERT INTO object_metadata
                        (object_id, name, category, image_paths, description)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (object_id, name, category, paths_json, description),
                )
                rowid = cursor.lastrowid

                # 再插入向量（rowid 关联元数据表）
                conn.execute(
                    """
                    INSERT INTO object_vectors (object_rowid, embedding)
                    VALUES (?, ?)
                    """,
                    (rowid, vec_str),
                )

                conn.commit()
                logger.info(
                    f"物体入库成功: object_id={object_id}, rowid={rowid}, "
                    f"dim={self.vector_dim}"
                )
                return rowid

            except sqlite3.IntegrityError as e:
                conn.rollback()
                if "UNIQUE" in str(e).upper():
                    raise ValueError(
                        f"物体 '{object_id}' 已存在，请使用不同的 object_id"
                    ) from e
                raise

    def update_object(
        self,
        object_id: str,
        embedding: np.ndarray,
        name: Optional[str] = None,
        category: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> bool:
        """
        更新已有物体的嵌入向量和元数据。

        参数:
            object_id:   物体唯一标识
            embedding:   新的归一化嵌入向量
            name:        新名称（None 则不更新）
            category:    新分类（None 则不更新）
            image_paths: 新图片路径列表（None 则不更新）
            description: 新描述（None 则不更新）

        返回:
            是否更新成功
        """
        if embedding.shape != (self.vector_dim,):
            raise ValueError(
                f"向量维度不匹配: 期望 ({self.vector_dim},)，实际 {embedding.shape}"
            )

        vec_str = _numpy_to_vec_string(embedding)

        with _DB_LOCK:
            conn = self._get_connection()
            # 获取 rowid
            row = conn.execute(
                "SELECT id FROM object_metadata WHERE object_id = ?",
                (object_id,),
            ).fetchone()
            if row is None:
                logger.warning(f"物体 '{object_id}' 不存在，无法更新")
                return False

            rowid = row[0]

            # 构建元数据更新语句
            updates: List[str] = []
            params: List[Any] = []
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if category is not None:
                updates.append("category = ?")
                params.append(category)
            if image_paths is not None:
                updates.append("image_paths = ?")
                params.append(json.dumps(image_paths, ensure_ascii=False))
            if description is not None:
                updates.append("description = ?")
                params.append(description)

            if updates:
                params.append(object_id)
                conn.execute(
                    f"UPDATE object_metadata SET {', '.join(updates)} WHERE object_id = ?",
                    params,
                )

            # 更新向量：先删后插（vec0 不支持 UPDATE）
            conn.execute(
                "DELETE FROM object_vectors WHERE object_rowid = ?", (rowid,)
            )
            conn.execute(
                "INSERT INTO object_vectors (object_rowid, embedding) VALUES (?, ?)",
                (rowid, vec_str),
            )

            conn.commit()
            logger.info(f"物体更新成功: object_id={object_id}")
            return True

    def delete_object(self, object_id: str) -> bool:
        """
        删除指定物体。

        参数:
            object_id: 物体唯一标识

        返回:
            是否删除成功
        """
        with _DB_LOCK:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT id FROM object_metadata WHERE object_id = ?",
                (object_id,),
            ).fetchone()
            if row is None:
                return False

            rowid = row[0]
            conn.execute(
                "DELETE FROM object_vectors WHERE object_rowid = ?", (rowid,)
            )
            conn.execute(
                "DELETE FROM object_metadata WHERE id = ?", (rowid,)
            )
            conn.commit()
            logger.info(f"物体已删除: object_id={object_id}")
            return True

    # ------------------------------------------------------------------ #
    #  查询操作
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        检索最相似的物体。

        参数:
            query_embedding: 归一化后的查询向量
            top_k:           返回结果数量上限

        返回:
            按距离升序排列的结果列表，每项包含:
            {
                "object_id": str,
                "name": str,
                "category": str,
                "distance": float,
                "image_paths": list[str],
                "description": str,
                "created_at": str,
            }
        """
        if query_embedding.shape != (self.vector_dim,):
            raise ValueError(
                f"查询向量维度不匹配: 期望 ({self.vector_dim},)，"
                f"实际 {query_embedding.shape}"
            )

        vec_str = _numpy_to_vec_string(query_embedding)
        conn = self._get_connection()

        # 使用 sqlite-vec 的 MATCH 语法做近似检索
        rows = conn.execute(
            """
            SELECT
                v.object_rowid,
                v.distance,
                m.object_id,
                m.name,
                m.category,
                m.image_paths,
                m.description,
                m.created_at
            FROM object_vectors v
            JOIN object_metadata m ON m.id = v.object_rowid
            WHERE v.embedding MATCH ?
            AND k = ?
            ORDER BY v.distance
            """,
            (vec_str, top_k),
        ).fetchall()

        results = []
        for row in rows:
            results.append({
                "object_id": row[2],
                "name": row[3],
                "category": row[4],
                "distance": float(row[1]),
                "image_paths": json.loads(row[5]) if row[5] else [],
                "description": row[6] or "",
                "created_at": row[7] or "",
            })

        logger.debug(f"向量搜索完成: top_k={top_k}, 返回 {len(results)} 条结果")
        return results

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 object_id 获取物体元数据。

        参数:
            object_id: 物体唯一标识

        返回:
            物体元数据字典，不存在则返回 None
        """
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT object_id, name, category, image_paths, description, created_at
            FROM object_metadata
            WHERE object_id = ?
            """,
            (object_id,),
        ).fetchone()

        if row is None:
            return None

        return {
            "object_id": row[0],
            "name": row[1],
            "category": row[2],
            "image_paths": json.loads(row[3]) if row[3] else [],
            "description": row[4] or "",
            "created_at": row[5] or "",
        }

    def list_objects(
        self,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        列出所有已存储的物体。

        参数:
            category: 可选分类过滤
            limit:    返回数量上限

        返回:
            物体元数据列表
        """
        conn = self._get_connection()
        if category:
            rows = conn.execute(
                """
                SELECT object_id, name, category, image_paths, description, created_at
                FROM object_metadata
                WHERE category = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT object_id, name, category, image_paths, description, created_at
                FROM object_metadata
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "object_id": r[0],
                "name": r[1],
                "category": r[2],
                "image_paths": json.loads(r[3]) if r[3] else [],
                "description": r[4] or "",
                "created_at": r[5] or "",
            }
            for r in rows
        ]

    def count(self) -> int:
        """返回数据库中物体总数"""
        conn = self._get_connection()
        row = conn.execute("SELECT COUNT(*) FROM object_metadata").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """关闭当前线程的数据库连接"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
            logger.debug("数据库连接已关闭")


# ====================================================================== #
#  辅助函数
# ====================================================================== #


def _numpy_to_vec_string(vec: np.ndarray) -> str:
    """
    将 numpy 向量转为 sqlite-vec 兼容的字符串格式。
    格式: "[0.123, -0.456, ...]"
    """
    return "[" + ",".join(f"{v:.8f}" for v in vec.astype(np.float32)) + "]"


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    将向量归一化为单位向量（L2 归一化）。
    归一化后可使用 L2 距离近似余弦相似度。

    参数:
        vec: 输入向量

    返回:
        单位向量（若输入为零向量则返回原向量）
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        logger.warning("输入零向量，跳过归一化")
        return vec
    return vec / norm


__all__ = [
    "VectorDB",
    "normalize_vector",
]
