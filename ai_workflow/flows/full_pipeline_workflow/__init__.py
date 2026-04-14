"""
全流程一键 Pipeline 工作流（LangGraph DAG）

将三个独立子工作流串联为单次调用：
  run_multi_scene → run_model_retrieval → run_scene_composition → END

用户只需发送一条 /full_pipeline 指令，即可完成：
  1. 设计方案分析与参考图生成（integrated_multi_scene_workflow）
  2. 3D 模型检索 / 生成（model_retrieval_workflow）
  3. 场景自动组合与导入（scene_composition_workflow）

子工作流之间通过 state.global_assets 传递数据：
  multi_scene   节点产出 → global_assets.multi_scene
  model_retrieval 节点消费 global_assets.multi_scene，产出 → global_assets.model_retrieval
  scene_composition 节点消费 global_assets.model_retrieval，产出 → global_assets.scene_composition
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from ai_workflow.executor import register_workflow_checkpoints
from ai_workflow.state import WorkflowState

from .constants import FULL_PIPELINE_FUNCTION_ID
from .nodes import (
    run_multi_scene_node,
    run_model_retrieval_node,
    run_scene_composition_node,
)

try:
    from .test_cases import TEST_CASES
except ImportError:
    TEST_CASES = {}

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_full_pipeline_workflow() -> "CompiledStateGraph":
    """构建全流程一键 Pipeline LangGraph DAG。"""
    graph = StateGraph(WorkflowState)

    graph.add_node("multi_scene", run_multi_scene_node)
    graph.add_node("model_retrieval", run_model_retrieval_node)
    graph.add_node("scene_composition", run_scene_composition_node)

    graph.add_edge(START, "multi_scene")
    graph.add_edge("multi_scene", "model_retrieval")
    graph.add_edge("model_retrieval", "scene_composition")
    graph.add_edge("scene_composition", END)

    return graph.compile()


WORKFLOWS: Dict[int, "CompiledStateGraph"] = {
    FULL_PIPELINE_FUNCTION_ID: build_full_pipeline_workflow(),
}

WORKFLOW_COMMANDS: Dict[str, int] = {
    "/full_pipeline": FULL_PIPELINE_FUNCTION_ID,
    "/pipeline": FULL_PIPELINE_FUNCTION_ID,
}

register_workflow_checkpoints(
    FULL_PIPELINE_FUNCTION_ID,
    {"multi_scene", "model_retrieval", "scene_composition"},
)

__all__ = [
    "TEST_CASES",
    "WORKFLOWS",
    "WORKFLOW_COMMANDS",
    "FULL_PIPELINE_FUNCTION_ID",
    "build_full_pipeline_workflow",
]
