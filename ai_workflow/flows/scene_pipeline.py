"""
场景生成主工作流（Pipeline）

将「多场景室内设计」与「模型检索/3D 生成」两步串联为完整 DAG：

  ┌── 第一步：方案设计 ──────────────────────────────────────────┐
  │ START → analyzer → human_review ─→ generate_images       ─┐ │
  │                                  └→ generate_layout_text ─┤ │
  │                                                  aggregate ─┘│
  └──────────────────────────────────────────────────────────────┘
                               │
  ┌── 第二步：模型获取 ──────────────────────────────────────────┐
  │              dispatch → retrieve_or_generate → register     │
  │                                               → format → END│
  └──────────────────────────────────────────────────────────────┘

对外使用 function_id = 21000 (SCENE_PIPELINE_FUNCTION_ID)。
"场景生成：" 前缀路由到此工作流。
"""

from __future__ import annotations

import logging
from typing import Dict, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from ai_workflow.state import WorkflowState

# 第一步节点
from ai_workflow.flows.integrated_multi_scene_workflow import (
    analyzer_node,
    human_review_node,
    generate_images_node,
    generate_layout_text_node,
    aggregate_result_node,
)

# 第二步节点
from ai_workflow.flows.model_retrieval_workflow import (
    dispatch_node,
    retrieve_or_generate_node,
    register_node,
    format_result_node,
)
from ai_workflow.executor import register_workflow_checkpoints

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

SCENE_PIPELINE_FUNCTION_ID = 21000


# ---------------------------------------------------------------------------
# DAG 构建与导出
# ---------------------------------------------------------------------------


def build_scene_pipeline() -> "CompiledStateGraph":
    """构建场景生成完整流水线 DAG。

    拓扑：
        START → analyzer → human_review ─→ generate_images       ─┐
                                         └→ generate_layout_text ─┤
                                                      aggregate_result
                                                           │
                                                       dispatch
                                                           │
                                                  retrieve_or_generate
                                                           │
                                                       register
                                                           │
                                                     format_result → END
    """
    graph = StateGraph(WorkflowState)

    # ---- 第一步节点 ----
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("generate_images", generate_images_node)
    graph.add_node("generate_layout_text", generate_layout_text_node)
    graph.add_node("aggregate_result", aggregate_result_node)

    # ---- 第二步节点 ----
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("retrieve_or_generate", retrieve_or_generate_node)
    graph.add_node("register", register_node)
    graph.add_node("format_result", format_result_node)

    # ---- 第一步边 ----
    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "human_review")
    graph.add_edge("human_review", "generate_images")
    graph.add_edge("human_review", "generate_layout_text")
    graph.add_edge("generate_images", "aggregate_result")
    graph.add_edge("generate_layout_text", "aggregate_result")

    # ---- 衔接：第一步 → 第二步 ----
    graph.add_edge("aggregate_result", "dispatch")

    # ---- 第二步边 ----
    graph.add_edge("dispatch", "retrieve_or_generate")
    graph.add_edge("retrieve_or_generate", "register")
    graph.add_edge("register", "format_result")
    graph.add_edge("format_result", END)

    return graph.compile()


WORKFLOWS: Dict[int, "CompiledStateGraph"] = {
    SCENE_PIPELINE_FUNCTION_ID: build_scene_pipeline(),
}

WORKFLOW_COMMANDS: Dict[str, int] = {
    "/scene": SCENE_PIPELINE_FUNCTION_ID,
    "/scene_pipeline": SCENE_PIPELINE_FUNCTION_ID,
}

register_workflow_checkpoints(
    SCENE_PIPELINE_FUNCTION_ID,
    {"aggregate_result", "format_result"},
)

__all__ = [
    "WORKFLOWS",
    "WORKFLOW_COMMANDS",
    "SCENE_PIPELINE_FUNCTION_ID",
    "build_scene_pipeline",
]
