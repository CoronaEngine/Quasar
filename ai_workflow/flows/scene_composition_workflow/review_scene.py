"""review_scene 节点 — 调用 VLM 场景合理性审查工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import get_tool, parse_review_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 审查决策分类
# ---------------------------------------------------------------------------

def _classify_review_decision(review_result: Dict[str, Any]) -> str:
    """根据审查结果分类决策：pass / layout_issue / object_issue。

    - pass        : 审查通过或跳过，无需重试
    - layout_issue: 物体本身没问题，摆放/朝向/间距不合理 → 重新走布局
    - object_issue: 模型本身质量差、类型错误、比例严重失调 → 重新生成物体
    """
    overall = review_result.get("overall", "PASS")
    if overall in {"PASS", "SKIPPED", "ERROR"}:
        return "pass"

    issues = review_result.get("issues", [])
    if not issues:
        return "layout_issue"

    issues_text = "\n".join(f"- {issue}" for issue in issues[:10])
    try:
        from ai_models.base_pool.registry import get_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_chat_model(temperature=0, request_timeout=15.0)
        response = llm.invoke([
            SystemMessage(content=(
                "你是一个场景审查问题分类器。根据以下问题列表判断主要问题类型：\n"
                "- layout_issue：物体位置、朝向、间距、摆放不合理等布局相关问题\n"
                "- object_issue：物体模型本身质量差、类型错误、比例严重失调、模型损坏等问题\n"
                "只输出 layout_issue 或 object_issue，不要输出其他任何内容。"
            )),
            HumanMessage(content=f"问题列表：\n{issues_text}"),
        ])
        text = (response.content if hasattr(response, "content") else str(response)).strip().lower()
        if "object_issue" in text:
            return "object_issue"
    except Exception as e:
        logger.warning("review_scene: 决策分类 LLM 调用失败，默认 layout_issue: %s", e)

    return "layout_issue"


@stream_output_node("integrated", NO_OUTPUT)
def review_scene_node(state) -> Dict[str, Any]:
    """调用 scene_rationality_review 对场景进行 VLM 质量审查。"""
    intermediate = state.get("intermediate", {})
    scene_json_path = intermediate.get("scene_json_path", "")
    scene_name = intermediate.get("scene_name", "composed_scene")
    prompt = state.get("prompt", "")
    current_retry = intermediate.get("review_retry_count", 0)

    tool = get_tool("scene_rationality_review")
    if tool is None:
        logger.warning("scene_rationality_review 工具未注册，跳过审查")
        return {
            "intermediate": {
                "review_result": {"overall": "SKIPPED", "score": -1, "issues": ["审查工具未注册"]},
                "review_decision": "pass",
            },
        }

    # 截图目录：优先使用 capture_screenshots 节点写入的路径，否则推断默认位置
    review_screenshot_dir = intermediate.get("review_screenshot_dir", "")
    if not review_screenshot_dir:
        output_dir = str(Path(scene_json_path).parent / "review_screenshots") if scene_json_path else ""
    else:
        output_dir = review_screenshot_dir
    scene_description = prompt or f"场景名: {scene_name}"

    # 目录不存在或没有 PNG 时直接跳过（--test 模式下无渲染截图）
    if not output_dir or not Path(output_dir).is_dir() or not any(
        p.name.lower().endswith(".png") for p in Path(output_dir).iterdir()
    ):
        logger.info("review_scene: 截图目录不存在或无 PNG，跳过审查 (output_dir=%s)", output_dir)
        return {
            "intermediate": {
                "review_result": {"overall": "SKIPPED", "score": -1, "issues": ["无截图，跳过审查"]},
                "review_decision": "pass",
            },
        }

    logger.info("review_scene: 调用 scene_rationality_review (output_dir=%s)", output_dir)

    try:
        raw_result = tool.invoke({
            "output_dir": output_dir,
            "scene_description": scene_description,
            "max_images": 12,
        })
        parsed = parse_review_result(raw_result)
        if parsed.get("error"):
            logger.warning("场景审查返回错误: %s", parsed["error"])
            return {
                "intermediate": {
                    "review_result": {"overall": "ERROR", "score": -1, "issues": [parsed["error"]]},
                    "review_decision": "pass",
                },
            }

        # 分类决策并更新重试计数
        decision = _classify_review_decision(parsed)
        new_retry_count = current_retry + 1

        # 若决策为 object_issue，标记需要重新生成模型（供 output_result 传递给外部系统）
        extra = {}
        if decision == "object_issue":
            extra["needs_model_regen"] = True

        logger.info(
            "review_scene: 审查完成 — overall=%s score=%s decision=%s (retry=%d→%d)",
            parsed.get("overall", "?"),
            parsed.get("score", "?"),
            decision,
            current_retry,
            new_retry_count,
        )
        return {
            "intermediate": {
                "review_result": parsed,
                "review_decision": decision,
                "review_retry_count": new_retry_count,
                **extra,
            },
        }

    except Exception as exc:
        logger.error("review_scene 异常: %s", exc, exc_info=True)
        return {
            "intermediate": {
                "review_result": {"overall": "ERROR", "score": -1, "issues": [str(exc)]},
                "review_decision": "pass",
            },
        }
