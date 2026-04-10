"""review_scene 节点 — 调用 VLM 场景合理性审查工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import get_tool, parse_review_result

logger = logging.getLogger(__name__)


@stream_output_node("integrated", NO_OUTPUT)
def review_scene_node(state) -> Dict[str, Any]:
    """调用 scene_rationality_review 对场景进行 VLM 质量审查。"""
    intermediate = state.get("intermediate", {})
    scene_json_path = intermediate.get("scene_json_path", "")
    scene_name = intermediate.get("scene_name", "composed_scene")
    prompt = state.get("prompt", "")

    tool = get_tool("scene_rationality_review")
    if tool is None:
        logger.warning("scene_rationality_review 工具未注册，跳过审查")
        return {
            "intermediate": {
                "review_result": {"overall": "SKIPPED", "score": -1, "issues": ["审查工具未注册"]},
            },
        }

    # 截图输出目录 = scene.json 所在目录下 /review_screenshots/
    output_dir = str(Path(scene_json_path).parent / "review_screenshots") if scene_json_path else ""
    scene_description = prompt or f"场景名: {scene_name}"

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
                },
            }

        logger.info(
            "review_scene: 审查完成 — %s (score=%s)",
            parsed.get("overall", "?"),
            parsed.get("score", "?"),
        )
        return {
            "intermediate": {
                "review_result": parsed,
            },
        }

    except Exception as exc:
        logger.error("review_scene 异常: %s", exc, exc_info=True)
        return {
            "intermediate": {
                "review_result": {"overall": "ERROR", "score": -1, "issues": [str(exc)]},
            },
        }
