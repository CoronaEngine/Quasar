"""
物体识别模块 —— 测试脚本

分层测试，由浅入深：
    1. 向量数据库读写（本地 CPU，秒级完成）
    2. 云端嵌入服务连通性（需要可访问的 embedding 服务）
    3. 端到端流程：远端入库 → 搜索

运行方式:
    cd d:/CodeLib/CoronaEngine/editor/CabbageEditor/plugins/AITool/CoronaArtificialIntelligence
    python -m ai_modules.object_recognition.test_recognition

或直接运行:
    python ai_modules/object_recognition/test_recognition.py
"""

from __future__ import annotations

import builtins
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ai_root = Path(__file__).resolve().parents[2]
editor_root = Path(__file__).resolve().parents[5]
for path in (ai_root, editor_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

if __package__ in (None, ""):
    __package__ = "ai_modules.object_recognition"

logger = logging.getLogger(__name__)


def print(*args, **kwargs):
    builtins.print(*args, **kwargs)

    file = kwargs.get("file", None)
    if file not in (None, sys.stdout, sys.stderr):
        return

    sep = kwargs.get("sep", " ")
    message = sep.join(str(a) for a in args).strip()
    if message:
        logger.info(message)


# ====================================================================== #
#  测试 1: 向量数据库（纯 CPU，无需模型）
# ====================================================================== #
def test_vector_db():
    """测试 sqlite-vec 向量数据库的增删查改"""
    print("\n" + "=" * 60)
    print("测试 1: 向量数据库 (sqlite-vec)")
    print("=" * 60)

    from .tools.vector_db import (
        VectorDB,
        normalize_vector,
    )

    # 使用临时文件，测试完自动清理
    db_path = os.path.join(tempfile.gettempdir(), "test_object_recognition.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = VectorDB(db_path=db_path, vector_dim=128)  # 用小维度加速测试
    print(f"  数据库路径: {db_path}")

    # ── 插入 ──
    vec_a = normalize_vector(np.random.randn(128).astype(np.float32))
    vec_b = normalize_vector(np.random.randn(128).astype(np.float32))

    db.insert_object(
        object_id="cup_001",
        embedding=vec_a,
        name="白色陶瓷杯",
        category="餐具",
        image_paths=["front.jpg", "back.jpg"],
        description="一个普通的白色陶瓷杯",
    )
    print("  插入 cup_001 ✓")

    db.insert_object(
        object_id="mouse_001",
        embedding=vec_b,
        name="黑色无线鼠标",
        category="电子设备",
        image_paths=["front.jpg", "back.jpg", "left.jpg"],
        description="一个黑色的无线鼠标",
    )
    print("  插入 mouse_001 ✓")

    # ── 计数 ──
    count = db.count()
    assert count == 2, f"期望 2 条记录，实际 {count}"
    print(f"  记录总数: {count} ✓")

    # ── 查询单条 ──
    obj = db.get_object("cup_001")
    assert obj is not None
    assert obj["name"] == "白色陶瓷杯"
    print(f"  查询 cup_001: {obj['name']} ✓")

    # ── 向量搜索 ──
    results = db.search(query_embedding=vec_a, top_k=2)
    assert len(results) >= 1
    assert results[0]["object_id"] == "cup_001"  # 自身应该是最相似的
    summary = [(r["object_id"], f"{r['distance']:.4f}") for r in results]
    print(f"  搜索结果: {summary} ✓")

    # ── 列表 ──
    all_objects = db.list_objects()
    assert len(all_objects) == 2
    print(f"  列出所有物体: {[o['object_id'] for o in all_objects]} ✓")

    # ── 分类过滤 ──
    filtered = db.list_objects(category="餐具")
    assert len(filtered) == 1
    print(f"  按分类过滤 '餐具': {[o['object_id'] for o in filtered]} ✓")

    # ── 更新 ──
    vec_a_new = normalize_vector(np.random.randn(128).astype(np.float32))
    ok = db.update_object(
        object_id="cup_001",
        embedding=vec_a_new,
        name="蓝色马克杯",
    )
    assert ok
    obj = db.get_object("cup_001")
    assert obj["name"] == "蓝色马克杯"
    print(f"  更新 cup_001 名称为 '{obj['name']}' ✓")

    # ── 删除 ──
    ok = db.delete_object("mouse_001")
    assert ok
    assert db.count() == 1
    print("  删除 mouse_001 ✓")

    # ── 重复插入应报错 ──
    try:
        db.insert_object(object_id="cup_001", embedding=vec_a_new)
        print("  ✗ 重复插入未报错")
    except ValueError as e:
        print(f"  重复插入正确报错: {e} ✓")

    db.close()
    os.remove(db_path)
    print("\n  向量数据库测试全部通过 ✓")


# ====================================================================== #
#  测试 2: 云端嵌入服务
# ====================================================================== #
def test_remote_embedding_service():
    """测试云端 embedding 服务的存储/查询接口"""
    print("\n" + "=" * 60)
    print("测试 2: 云端嵌入服务")
    print("=" * 60)

    from .configs.dataclasses import RecognitionConfig
    from .tools.client_embedding import get_embedding_client

    config = RecognitionConfig()
    print(f"  远端服务: {config.remote_endpoint}")
    print(f"  输出维度: {config.embedding.output_dim}")
    client = get_embedding_client(
        endpoint=config.remote_endpoint,
        timeout=config.remote_timeout,
        output_dim=config.embedding.output_dim,
    )

    print("\n  [纯文本] 生成嵌入中...")
    try:
        vec_text = client.embed_for_storage(image_paths=[], text="一个白色的陶瓷杯子")
    except Exception as e:
        print(f"  跳过: 云端 embedding 服务不可用: {e}")
        return

    print(f"  向量维度: {vec_text.shape}")
    print(f"  范数 (应接近 1.0): {np.linalg.norm(vec_text):.6f}")
    assert vec_text.shape == (config.embedding.output_dim,)

    print("\n  [查询文本] 生成嵌入中...")
    vec_query = client.embed_for_query(text="杯子")
    print(f"  向量维度: {vec_query.shape}")

    cos_sim = np.dot(vec_text, vec_query)
    print(f"  '白色陶瓷杯子' vs '杯子' 余弦相似度: {cos_sim:.4f}")

    vec_unrelated = client.embed_for_query(text="一双红色运动鞋")
    cos_sim_unrelated = np.dot(vec_text, vec_unrelated)
    print(f"  '白色陶瓷杯子' vs '红色运动鞋' 余弦相似度: {cos_sim_unrelated:.4f}")

    assert cos_sim > cos_sim_unrelated, "相关文本的相似度应高于不相关文本"
    print("  语义区分验证通过 ✓")

    print("\n  云端嵌入服务测试通过 ✓")


# ====================================================================== #
#  测试 3: 端到端（入库 + 搜索）
# ====================================================================== #
def test_end_to_end():
    """端到端测试：通过云端 embedding 服务完成入库和搜索"""
    print("\n" + "=" * 60)
    print("测试 3: 端到端流程（入库 → 搜索）")
    print("=" * 60)

    from .configs.dataclasses import (
        RecognitionConfig,
        VectorDBConfig,
    )
    from .tools.client_embedding import build_provider
    from .tools.vector_db import VectorDB

    db_path = os.path.join(tempfile.gettempdir(), "test_e2e_recognition.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    recognition_cfg = RecognitionConfig(
        enable=True,
        vector_db=VectorDBConfig(db_path=db_path, vector_dim=1024),
    )
    db_cfg = recognition_cfg.vector_db

    client = build_provider(recognition_cfg)
    db = VectorDB(db_path=db_cfg.db_path, vector_dim=db_cfg.vector_dim)

    # ── 入库：3 个物体（纯文本，不需要图片文件） ──
    objects = [
        ("cup_001", "白色陶瓷杯", "餐具", "一个普通的白色陶瓷杯子，表面光滑"),
        ("mouse_001", "黑色无线鼠标", "电子设备", "一个黑色的无线鼠标，带有侧键"),
        ("chair_001", "办公椅", "家具", "一把黑色皮质办公转椅，带扶手和滚轮"),
    ]

    for obj_id, name, category, desc in objects:
        print(f"\n  [入库] {obj_id}: {name}")
        try:
            embedding = client.embed_for_storage(image_paths=[], text=desc)
        except Exception as e:
            db.close()
            os.remove(db_path)
            print(f"  跳过: 云端 embedding 服务不可用: {e}")
            return
        db.insert_object(
            object_id=obj_id,
            embedding=embedding,
            name=name,
            category=category,
            description=desc,
        )
        print(f"    向量维度: {embedding.shape}, 已存储 ✓")

    print(f"\n  数据库中共 {db.count()} 个物体")

    queries = [
        ("杯子", "cup_001"),
        ("鼠标", "mouse_001"),
        ("椅子", "chair_001"),
        ("一把办公用的椅子，有扶手", "chair_001"),
    ]

    print("\n  搜索测试:")
    all_correct = True
    for query_text, expected_id in queries:
        query_vec = client.embed_for_query(text=query_text)
        results = db.search(query_embedding=query_vec, top_k=3)
        top_result = results[0] if results else None

        if top_result and top_result["object_id"] == expected_id:
            status = "✓"
        else:
            status = "✗"
            all_correct = False

        print(f"    查询 '{query_text}':")
        for i, r in enumerate(results):
            marker = "→" if i == 0 else " "
            print(
                f"      {marker} {r['object_id']} | {r['name']} | "
                f"距离: {r['distance']:.4f}"
            )
        print(f"    预期: {expected_id}  {status}")

    db.close()
    os.remove(db_path)

    if all_correct:
        print("\n  端到端测试全部通过 ✓")
    else:
        print("\n  部分搜索结果不符合预期，请检查")


# ====================================================================== #
#  测试 4: 目录自动扫描（纯 CPU，无需模型）
# ====================================================================== #
def test_auto_scan():
    """测试目录自动扫描入库功能"""
    print("\n" + "=" * 60)
    print("测试 4: 目录自动扫描 (auto_scan)")
    print("=" * 60)

    import shutil

    from .configs.dataclasses import (
        RecognitionConfig,
        VectorDBConfig,
    )
    from .tools.auto_scan import scan_and_register
    from .tools.vector_db import (
        VectorDB,
        normalize_vector,
    )

    # ── 准备临时目录结构 ──
    base_dir = os.path.join(tempfile.gettempdir(), "test_auto_scan")
    scan_dir = os.path.join(base_dir, "objects")
    db_path = os.path.join(base_dir, "test_scan.db")

    # 清理旧数据
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(scan_dir)

    # 创建子文件夹：cup_001（有图片）、chair_001（有图片）、empty_box（无图片）
    cup_dir = os.path.join(scan_dir, "cup_001")
    chair_dir = os.path.join(scan_dir, "chair_001")
    empty_dir = os.path.join(scan_dir, "empty_box")
    hidden_dir = os.path.join(scan_dir, ".hidden")
    os.makedirs(cup_dir)
    os.makedirs(chair_dir)
    os.makedirs(empty_dir)
    os.makedirs(hidden_dir)

    # 写入假图片文件（内容不重要，扫描只检查扩展名）
    from PIL import Image

    for name in ["front.jpg", "back.jpg", "left.png"]:
        img = Image.new("RGB", (64, 64), color="white")
        img.save(os.path.join(cup_dir, name))
    for name in ["front.jpg", "side.jpg"]:
        img = Image.new("RGB", (64, 64), color="gray")
        img.save(os.path.join(chair_dir, name))
    # hidden 目录也放图片（应被忽略）
    img = Image.new("RGB", (64, 64), color="black")
    img.save(os.path.join(hidden_dir, "secret.jpg"))

    print(f"  临时目录: {scan_dir}")
    print(
        "  子文件夹: cup_001 (3 图), chair_001 (2 图), empty_box (0 图), .hidden (1 图)"
    )

    dim = 128
    db = VectorDB(db_path=db_path, vector_dim=dim)

    class MockEmbeddingClient:
        """用随机向量模拟云端嵌入，避免依赖远端服务。"""

        def embed_for_storage(self, image_paths, text="", instruction=None):
            return normalize_vector(np.random.randn(dim).astype(np.float32))

    mock_client = MockEmbeddingClient()

    # ── 场景 A: 开关关闭 (auto_scan_embed=False) → 只警告不入库 ──
    print("\n  [场景 A] auto_scan_embed=False → 仅警告")
    cfg_warn = RecognitionConfig(
        enable=True,
        vector_db=VectorDBConfig(db_path=db_path, vector_dim=dim),
        auto_scan_dir=scan_dir,
        auto_scan_embed=False,
        auto_scan_max_images=6,
    )

    stats_a = scan_and_register(cfg_warn, db, mock_client)
    print(f"    统计: {stats_a}")
    assert stats_a["scanned"] == 3, f"期望扫描 3 个子文件夹，实际 {stats_a['scanned']}"
    assert stats_a["warned"] == 2, f"期望警告 2 (cup+chair)，实际 {stats_a['warned']}"
    assert stats_a["skipped"] == 1, f"期望跳过 1 (empty)，实际 {stats_a['skipped']}"
    assert (
        stats_a["registered"] == 0
    ), f"开关关闭时不应入库，实际 {stats_a['registered']}"
    assert db.count() == 0, f"数据库应为空，实际 {db.count()}"
    print("    验证通过: 未入库，仅输出警告 ✓")

    # ── 场景 B: 开关开启 (auto_scan_embed=True) → 自动入库 ──
    print("\n  [场景 B] auto_scan_embed=True → 自动入库")
    cfg_embed = RecognitionConfig(
        enable=True,
        vector_db=VectorDBConfig(db_path=db_path, vector_dim=dim),
        auto_scan_dir=scan_dir,
        auto_scan_embed=True,
        auto_scan_max_images=6,
    )

    stats_b = scan_and_register(cfg_embed, db, mock_client)
    print(f"    统计: {stats_b}")
    assert stats_b["scanned"] == 3, f"期望扫描 3，实际 {stats_b['scanned']}"
    assert (
        stats_b["registered"] == 2
    ), f"期望入库 2 (cup+chair)，实际 {stats_b['registered']}"
    assert stats_b["skipped"] == 1, f"期望跳过 1 (empty)，实际 {stats_b['skipped']}"
    assert len(stats_b["errors"]) == 0, f"不应有错误，实际 {stats_b['errors']}"
    assert db.count() == 2, f"数据库应有 2 条记录，实际 {db.count()}"

    # 验证入库内容
    cup = db.get_object("cup_001")
    assert cup is not None
    assert cup["name"] == "cup_001"
    assert len(cup["image_paths"]) == 3
    print(f"    cup_001: name={cup['name']}, images={len(cup['image_paths'])} ✓")

    chair = db.get_object("chair_001")
    assert chair is not None
    assert len(chair["image_paths"]) == 2
    print(f"    chair_001: name={chair['name']}, images={len(chair['image_paths'])} ✓")
    print("    验证通过: 自动入库成功 ✓")

    # ── 场景 C: 再次扫描 → 已登记的应跳过 ──
    print("\n  [场景 C] 重复扫描 → 已登记的应跳过")
    stats_c = scan_and_register(cfg_embed, db, mock_client)
    print(f"    统计: {stats_c}")
    assert (
        stats_c["already_registered"] == 2
    ), f"期望已登记 2，实际 {stats_c['already_registered']}"
    assert stats_c["registered"] == 0, f"不应新增入库，实际 {stats_c['registered']}"
    assert db.count() == 2, f"数据库仍应有 2 条记录，实际 {db.count()}"
    print("    验证通过: 重复扫描正确跳过 ✓")

    # ── 场景 D: max_images 限制 ──
    print("\n  [场景 D] auto_scan_max_images=1 → 每个文件夹最多 1 张图")
    # 先删除已有记录重新测试
    db.delete_object("cup_001")
    db.delete_object("chair_001")

    cfg_limit = RecognitionConfig(
        enable=True,
        vector_db=VectorDBConfig(db_path=db_path, vector_dim=dim),
        auto_scan_dir=scan_dir,
        auto_scan_embed=True,
        auto_scan_max_images=1,
    )
    print(f"    统计: {scan_and_register(cfg_limit, db, mock_client)}")
    cup = db.get_object("cup_001")
    assert cup is not None
    assert (
        len(cup["image_paths"]) == 1
    ), f"max_images=1 时应只取 1 张，实际 {len(cup['image_paths'])}"
    print(f"    cup_001 图片数: {len(cup['image_paths'])} ✓")
    print("    验证通过: max_images 限制生效 ✓")

    # ── 清理 ──
    db.close()
    shutil.rmtree(base_dir)
    print("\n  目录自动扫描测试全部通过 ✓")


# ====================================================================== #
#  主入口
# ====================================================================== #
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    print("=" * 60)
    print("  物体识别模块 (object_recognition) 测试")
    print("=" * 60)

    # 测试 1: 向量数据库（无需 GPU，推荐先跑这个）
    try:
        test_vector_db()
    except Exception as e:
        print(f"\n  ✗ 向量数据库测试失败: {e}")
        import traceback

        traceback.print_exc()
        print("\n  提示: pip install sqlite-vec numpy")

    # 测试 2: 云端嵌入服务
    try:
        test_remote_embedding_service()
    except Exception as e:
        print(f"\n  ✗ 云端嵌入服务测试失败: {e}")
        import traceback

        traceback.print_exc()
        print("\n  提示: 确认 remote_endpoint 可访问，且已安装 httpx")

    # 测试 3: 端到端
    try:
        test_end_to_end()
    except Exception as e:
        print(f"\n  ✗ 端到端测试失败: {e}")
        import traceback

        traceback.print_exc()

    # 测试 4: 目录自动扫描（无需 GPU）
    try:
        test_auto_scan()
    except Exception as e:
        print(f"\n  ✗ 目录自动扫描测试失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("  全部测试完成")
    print("=" * 60)
