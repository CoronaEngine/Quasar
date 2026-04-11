"""scene_composition 工作流测试入口。

使用方法:
    在引擎 Python 控制台或脚本中调用:

    from ai_workflow.flows.scene_composition_workflow.test_cases import run_test
    run_test()

    # 或自定义模型列表:
    run_test(models=[
        {"name": "椅子", "path": "D:/project/test/New_Corona_Project/models/base_basic_pbr_10/椅子.glb"},
        {"name": "桌子", "path": "D:/project/test/New_Corona_Project/models/base_basic_pbr_11/桌子.glb"},
    ])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 默认测试模型（修改为你本地实际存在的路径） ──────────────────────────

DEFAULT_MODELS: List[Dict[str, str]] = [
    {"name": "不对称环形吊灯", "path": "D:/project/test/New_Corona_Project/models/base_basic_pbr_9/不对称环形吊灯.glb"},
    {"name": "可调光蚕丝落地灯", "path": "D:/project/test/New_Corona_Project/models/base_basic_pbr_20/可调光蚕丝落地灯.glb"},
    {"name": "不对称亚克力悬浮床头柜", "path": "D:/project/test/New_Corona_Project/models/base_basic_pbr_19/不对称亚克力悬浮床头柜.glb"},
    {"name": "弧形编织藤条休闲椅", "path": "D:/project/test/New_Corona_Project/models/base_basic_pbr_11/弧形编织藤条休闲椅.glb"},
    {"name": "悬浮式胡桃木平台床", "path": "D:/project/test/New_Corona_Project/models/base_basic_pbr_14/悬浮式胡桃木平台床.glb"},
    
]


def build_test_state(
    models: Optional[List[Dict[str, str]]] = None,
    *,
    session_id: str = "test-scene-composition",
    scene_name: str = "test_scene",
    room_size: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """构造可直接传入 scene_composition 工作流的初始 state。

    Args:
        models: 模型列表，每项需包含 ``name`` 和 ``path``。
        session_id: 会话 ID。
        scene_name: 输出场景名称。
        room_size: 房间尺寸 [L, W, H]，默认 [10, 3, 10]。
    """
    items = models or DEFAULT_MODELS
    if not items:
        raise ValueError(
            "models 为空，请传入至少一个模型。示例:\n"
            '  run_test(models=[{"name": "椅子", "path": "D:/path/to/椅子.glb"}])'
        )

    model_results = []
    for i, m in enumerate(items, 1):
        model_results.append({
            "item_name": m["name"],
            "object_id": m.get("object_id", m["name"]),
            "task_index": i,
            "source": "generation",
            "model_path": m["path"],
            "review_passed": True,
        })

    return {
        "session_id": session_id,
        "function_id": 21003,
        "prompt": "",
        "global_assets": { 
            "model_retrieval": {
                "model_results": model_results,
            },
        },
        "intermediate": {},
        "metadata": {
            "scene_name": scene_name,
            "room_size": room_size or [10, 3, 10],
        },
    }


def run_test(
    models: Optional[List[Dict[str, str]]] = None,
    *,
    session_id: str = "test-scene-composition",
    scene_name: str = "test_scene",
    room_size: Optional[List[float]] = None,
    stream: bool = False,
) -> Any:
    """直接执行 scene_composition 工作流。

    Args:
        models: 模型列表，每项 ``{"name": "...", "path": "..."}``。
        session_id: 会话 ID。
        scene_name: 输出场景名称。
        room_size: 房间尺寸。
        stream: 是否使用流式执行（打印中间输出）。

    Returns:
        工作流最终 state（非流式）或 None（流式，结果打印到日志）。
    """
    state = build_test_state(
        models,
        session_id=session_id,
        scene_name=scene_name,
        room_size=room_size,
    )

    from . import build_scene_composition_workflow

    graph = build_scene_composition_workflow()

    if stream:
        logger.info("=== scene_composition 流式测试开始 ===")
        for chunk in graph.stream(state, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                error = node_update.get("error") if isinstance(node_update, dict) else None
                if error:
                    logger.error("[%s] 错误: %s", node_name, error)
                else:
                    logger.info("[%s] 完成", node_name)
        logger.info("=== scene_composition 流式测试结束 ===")
        return None

    logger.info("=== scene_composition 测试开始 ===")
    final_state = graph.invoke(state)
    error = final_state.get("error")
    if error:
        logger.error("工作流失败: %s", error)
    else:
        intermediate = final_state.get("intermediate", {})
        scene_path = intermediate.get("scene_json_path", "未知")
        imported = intermediate.get("imported_actors", [])
        failed = intermediate.get("failed_actors", [])
        logger.info(
            "工作流完成: scene_path=%s, 导入成功=%d, 导入失败=%d",
            scene_path, len(imported), len(failed),
        )
    logger.info("=== scene_composition 测试结束 ===")
    return final_state
