"""
第三步工作流：场景组合（LangGraph DAG）

接收第二步（模型检索与 3D 生成工作流）的输出状态，执行：
  1. collect_models — 从 global_assets.model_retrieval 提取可用模型列表
  2. compose_scene — 调用 place_scene_from_items 生成 scene.json 布局
  3. import_to_engine — 将 actor 逐一导入运行中的引擎场景
  4. review_scene — 调用 VLM 对场景进行合理性审查
  5. output_result — 汇总结果写入 global_assets.scene_composition

DAG 拓扑：
  START → collect_models → compose_scene → import_to_engine
       → review_scene → output_result → END

保持对外接口约定（function_id、WORKFLOWS / WORKFLOW_COMMANDS 导出）。
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from ai_workflow.executor import register_workflow_checkpoints
from ai_workflow.state import SceneCompositionWorkflowState

from .collect_models import collect_models_node
from .compose_scene import compose_scene_node
from .constants import SCENE_COMPOSITION_FUNCTION_ID
from .import_to_engine import import_to_engine_node
from .output_result import output_result_node
from .review_scene import review_scene_node

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_scene_composition_workflow() -> "CompiledStateGraph":
    """构建场景组合 LangGraph DAG。"""
    graph = StateGraph(SceneCompositionWorkflowState)

    graph.add_node("collect_models", collect_models_node)
    graph.add_node("compose_scene", compose_scene_node)
    graph.add_node("import_to_engine", import_to_engine_node)
    graph.add_node("review_scene", review_scene_node)
    graph.add_node("output_result", output_result_node)

    graph.add_edge(START, "collect_models")
    graph.add_edge("collect_models", "compose_scene")
    graph.add_edge("compose_scene", "import_to_engine")
    graph.add_edge("import_to_engine", "review_scene")
    graph.add_edge("review_scene", "output_result")
    graph.add_edge("output_result", END)

    graph.set_entry_point("collect_models")
    graph.set_finish_point("output_result")

    return graph.compile()


WORKFLOWS: Dict[int, "CompiledStateGraph"] = {
    SCENE_COMPOSITION_FUNCTION_ID: build_scene_composition_workflow(),
}

WORKFLOW_COMMANDS: Dict[str, int] = {
    "/scene_composition": SCENE_COMPOSITION_FUNCTION_ID,
}

register_workflow_checkpoints(
    SCENE_COMPOSITION_FUNCTION_ID,
    {"compose_scene", "output_result"},
)

__all__ = [
    "WORKFLOWS",
    "WORKFLOW_COMMANDS",
    "SCENE_COMPOSITION_FUNCTION_ID",
    "build_scene_composition_workflow",
]
