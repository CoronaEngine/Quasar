"""scene_composition 工作流测试入口。

使用方法:
    在 Agent 对话框直接输入：

        /scene_composition --test

    或在引擎 Python 控制台调用：

        from ai_workflow.flows.scene_composition_workflow.test_cases import run_test
        run_test(stream=True)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 默认测试模型（修改为你本地实际存在的路径） ──────────────────────────

BasePath = "D:/CodeLib/New_Corona_Project/"

DEFAULT_MODELS: List[Dict[str, str]] = [
    {"name": "欧式双人床", "path": f"{BasePath}models/欧式双人床/欧式双人床.glb"},
    {"name": "现代沙发", "path": f"{BasePath}models/现代沙发/现代沙发.glb"},
    {"name": "艺术落地灯", "path": f"{BasePath}models/艺术落地灯/艺术落地灯.glb"},
    {"name": "教堂", "path": f"{BasePath}models/教堂/教堂.glb"},
    {"name": "红色复古双门轿跑", "path": f"{BasePath}models/红色复古双门轿跑/红色复古双门轿跑.glb"},
]

# 默认设计方案：与 DEFAULT_MODELS 中的模型对应
DEFAULT_PROMPT = (
    "现代展厅场景。房间尺寸为 10m（X）×10m（Z），天花板高度 3m。"
    "欧式双人床放在房间北侧中心，床头靠北墙。"
    "现代沙发放在房间中部偏南区域，朝向北侧。"
    "艺术落地灯放在沙发右后方（东侧），用于氛围照明。"
    "教堂模型放在房间西北角，作为背景主体景观。"
    "红色复古双门轿跑放在房间东南侧展示位，车头朝向西北方向。"
)


def build_test_state(
    models: Optional[List[Dict[str, str]]] = None,
    *,
    session_id: str = "test-scene-composition",
    scene_name: str = "test_scene",
    room_size: Optional[List[float]] = None,
    prompt: str = "",
) -> Dict[str, Any]:
    """构造可直接传入 scene_composition 工作流的初始 state。

    Args:
        models: 模型列表，每项需包含 ``name`` 和 ``path``。
        session_id: 会话 ID。
        scene_name: 输出场景名称。
        room_size: 房间尺寸 [X_length, Z_depth, Y_height]，默认 [10, 10, 3]。
        prompt: 设计方案描述，非空时触发 LLM 智能布局，为空则使用等间距默认布局。
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
        "prompt": prompt,
        "global_assets": {
            "model_retrieval": {
                "model_results": model_results,
            },
        },
        "intermediate": {},
        "metadata": {
            "scene_name": scene_name,
            "room_size": room_size or [10, 10, 3],
        },
    }


def run_test(
    models: Optional[List[Dict[str, str]]] = None,
    *,
    session_id: str = "test-scene-composition",
    scene_name: str = "test_scene",
    room_size: Optional[List[float]] = None,
    prompt: str = "",
    stream: bool = False,
) -> Any:
    """直接执行 scene_composition 工作流。

    Args:
        models: 模型列表，每项 ``{"name": "...", "path": "..."}``。
        session_id: 会话 ID。
        scene_name: 输出场景名称。
        room_size: 房间尺寸 [X_length, Z_depth, Y_height]。
        prompt: 设计方案描述，非空时触发 LLM 智能布局，为空则自动使用 DEFAULT_PROMPT。
        stream: 是否使用流式执行（打印中间输出）。

    Returns:
        工作流最终 state（非流式）或 None（流式，结果打印到日志）。
    """
    state = build_test_state(
        models,
        session_id=session_id,
        scene_name=scene_name,
        room_size=room_size,
        prompt=prompt or DEFAULT_PROMPT,
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
