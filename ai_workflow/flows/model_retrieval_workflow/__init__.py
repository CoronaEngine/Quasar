"""
第二步工作流：模型检索与 3D 生成（LangGraph DAG）

接收第一步（多场景室内设计工作流）的输出状态，对每个物体：
  1. 使用 object_recognition 模块检索已有 3D 模型
  2. 若检索命中（distance < 阈值），记录模型 ID
  3. 若未命中，调用 three_d_generate 模块生成新 3D 模型

DAG 拓扑：
  START → dispatch_node → retrieve_or_generate_node → register_node
      → format_result_node → END

保持对外接口约定（function_id、WORKFLOWS / WORKFLOW_COMMANDS 导出）。
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from ai_workflow.executor import register_workflow_checkpoints
from ai_workflow.state import ModelRetrievalWorkflowState

from .visual_review import visual_review_node
from .constants import MODEL_RETRIEVAL_FUNCTION_ID
from .dispatch import dispatch_node
from .format_result import format_result_node
from .register import register_node
from .retrieve_or_generate import retrieve_or_generate_node
from .six_view_capture_tool import six_view_capture_tool_node

try:
    from .test_cases import TEST_CASES
except ImportError:
    TEST_CASES = {}

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

def check_if_needs_retry(state: ModelRetrievalWorkflowState) -> str:
    """动态路由决策器"""
    if state.get("needs_retry"):
        return "retrieve_or_generate"  # 如果有模型不合格，回到生成节点重做
    return "format_result"             # 全员合格，进入最后输出

def build_model_retrieval_workflow() -> "CompiledStateGraph":
    """构建模型检索与生成 LangGraph DAG。"""
    graph = StateGraph(ModelRetrievalWorkflowState)

    graph.add_node("dispatch", dispatch_node)
    graph.add_node("retrieve_or_generate", retrieve_or_generate_node)
    graph.add_node("register", register_node)
    graph.add_node("capture_views", six_view_capture_tool_node)
    graph.add_node("visual_review", visual_review_node)
    graph.add_node("format_result", format_result_node)

    graph.add_edge(START, "dispatch")
    graph.add_edge("dispatch", "retrieve_or_generate")
    graph.add_edge("retrieve_or_generate", "register")
    graph.add_edge("register", "capture_views")  
    graph.add_edge("capture_views", "visual_review")

    graph.add_conditional_edges(
        "visual_review",
        check_if_needs_retry,
        {
            "retrieve_or_generate": "retrieve_or_generate", # 回滚
            "format_result": "format_result"                # 通关
        }
    )
    
    graph.add_edge("format_result", END)
    
    graph.set_entry_point("dispatch")
    graph.set_finish_point("format_result")
    
    return graph.compile()


WORKFLOWS: Dict[int, "CompiledStateGraph"] = {
    MODEL_RETRIEVAL_FUNCTION_ID: build_model_retrieval_workflow(),
}

WORKFLOW_COMMANDS: Dict[str, int] = {
    "/model_retrieval": MODEL_RETRIEVAL_FUNCTION_ID,
}

register_workflow_checkpoints(
    MODEL_RETRIEVAL_FUNCTION_ID,
    {"retrieve_or_generate", "format_result"},
)

__all__ = [
    "WORKFLOWS",
    "WORKFLOW_COMMANDS",
    "MODEL_RETRIEVAL_FUNCTION_ID",
    "build_model_retrieval_workflow",
    "TEST_CASES",
]
