"""
全流程 Pipeline 各阶段节点

每个节点完整地调用一个子工作流，并将子工作流产出的
global_assets / dialogue_entries 回写到 pipeline state，
由 LangGraph 的 reducer (deep_merge_dict / operator.add) 自动累积传递。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ai_workflow.state import (
    MultiSceneWorkflowState,
    ModelRetrievalWorkflowState,
    SceneCompositionWorkflowState,
    deep_merge_dict,
)

_logger = logging.getLogger(__name__)


def _make_sub_state(pipeline_state: Dict[str, Any], function_id: int) -> Dict[str, Any]:
    """从 pipeline state 衍生子工作流初始状态。

    直接将当前已积累的 global_assets 注入子工作流，使其能读到上一步的产出。

    resume_from_review 路径说明：
      parse_request 在审核提交后会将 approved_elements 写入顶层 state，
      同时在 metadata 中置 resume_from_review=True。
      这两个字段必须一起透传给子工作流，否则 analyzer/human_review 节点
      虽看到 resume 标记但找不到元素，会重新调用 LLM。
    """
    return {
        "session_id": pipeline_state.get("session_id", "default"),
        "function_id": function_id,
        "prompt": pipeline_state.get("prompt", ""),
        "images": pipeline_state.get("images", []),
        "additional_type": pipeline_state.get("additional_type", []),
        "bounding_box": pipeline_state.get("bounding_box", []),
        "resolution": pipeline_state.get("resolution", "1:1"),
        "image_size": pipeline_state.get("image_size", "2K"),
        "metadata": dict(pipeline_state.get("metadata", {})),
        # 关键：把上一步积累的 global_assets 整体传入
        "global_assets": deep_merge_dict({}, pipeline_state.get("global_assets", {})),
        # resume_from_review 时透传已审核的元素列表，供子工作流跳过 analyzer/human_review
        "approved_elements": list(pipeline_state.get("approved_elements", []) or []),
        "extracted_elements": list(pipeline_state.get("extracted_elements", []) or []),
        "dialogue_entries": [],
        "intermediate": {},
        "error": None,
    }


def run_multi_scene_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """阶段 1/3：多物体场景设计分析，产出设计方案与参考图。"""
    # 延迟导入避免循环依赖；复用预构建图实例，不重复编译
    from ai_workflow.flows.integrated_multi_scene_workflow import (
        WORKFLOWS as _MS_WORKFLOWS,
        MULTI_SCENE_FUNCTION_ID,
    )

    _logger.info("[Pipeline] ▶ 阶段 1/3 multi_scene_workflow 开始")

    sub_state: MultiSceneWorkflowState = _make_sub_state(state, MULTI_SCENE_FUNCTION_ID)  # type: ignore[assignment]
    graph = _MS_WORKFLOWS[MULTI_SCENE_FUNCTION_ID]
    final = graph.invoke(sub_state)

    _logger.info(
        "[Pipeline] ✔ 阶段 1/3 完成，approved_elements=%d",
        len(final.get("global_assets", {}).get("multi_scene", {}).get("approved_elements", [])),
    )

    return {
        "global_assets": final.get("global_assets", {}),
        "dialogue_entries": final.get("dialogue_entries", []),
    }


def run_model_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """阶段 2/3：模型检索与 3D 生成，为每个设计元素生成/检索 3D 模型。"""
    from ai_workflow.flows.model_retrieval_workflow import (
        WORKFLOWS as _MR_WORKFLOWS,
        MODEL_RETRIEVAL_FUNCTION_ID,
    )

    _logger.info("[Pipeline] ▶ 阶段 2/3 model_retrieval_workflow 开始")

    sub_state: ModelRetrievalWorkflowState = _make_sub_state(state, MODEL_RETRIEVAL_FUNCTION_ID)  # type: ignore[assignment]
    graph = _MR_WORKFLOWS[MODEL_RETRIEVAL_FUNCTION_ID]
    final = graph.invoke(sub_state)

    model_results = final.get("global_assets", {}).get("model_retrieval", {}).get("model_results", [])
    _logger.info("[Pipeline] ✔ 阶段 2/3 完成，model_results=%d", len(model_results))

    return {
        "global_assets": final.get("global_assets", {}),
        "dialogue_entries": final.get("dialogue_entries", []),
    }


def run_scene_composition_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """阶段 3/3：场景组合，将 3D 模型导入并编排最终场景。"""
    from ai_workflow.flows.scene_composition_workflow import (
        WORKFLOWS as _SC_WORKFLOWS,
        SCENE_COMPOSITION_FUNCTION_ID,
    )

    _logger.info("[Pipeline] ▶ 阶段 3/3 scene_composition_workflow 开始")

    sub_state: SceneCompositionWorkflowState = _make_sub_state(state, SCENE_COMPOSITION_FUNCTION_ID)  # type: ignore[assignment]
    graph = _SC_WORKFLOWS[SCENE_COMPOSITION_FUNCTION_ID]
    final = graph.invoke(sub_state)

    scene_path = final.get("global_assets", {}).get("scene_composition", {}).get("scene_path", "")
    _logger.info("[Pipeline] ✔ 阶段 3/3 完成，scene_path=%s", scene_path)

    return {
        "global_assets": final.get("global_assets", {}),
        "dialogue_entries": final.get("dialogue_entries", []),
    }
