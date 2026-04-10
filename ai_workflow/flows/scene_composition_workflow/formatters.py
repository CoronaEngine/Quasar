from __future__ import annotations

from typing import Any, Dict, List

from ai_workflow.streaming import FormatterFunc

NO_OUTPUT: FormatterFunc = lambda _data, _state: []


def format_composition_result_parts(data: Dict[str, Any], state) -> List[Dict[str, Any]]:
    """为场景组合结果输出可视化摘要。"""
    scene_path = data.get("scene_path", "")
    imported = data.get("imported_actors", [])
    failed = data.get("failed_actors", [])

    if not scene_path and not imported:
        return []

    lines = ["## 场景组合结果"]
    if scene_path:
        lines.append(f"场景文件: `{scene_path}`")
    lines.append(f"成功导入: **{len(imported)}** 个模型")
    if failed:
        lines.append(f"导入失败: **{len(failed)}** 个模型")
    lines.append("")

    for actor in imported:
        lines.append(f"- ✅ {actor.get('name', '未知')}")
    for actor in failed:
        lines.append(f"- ❌ {actor.get('name', '未知')}: {actor.get('error', '')}")

    review = data.get("review_result", {})
    if review:
        overall = review.get("overall", "N/A")
        score = review.get("score", "N/A")
        lines.append("")
        lines.append(f"### 场景审查: {overall} (评分: {score})")
        for issue in review.get("issues", []):
            lines.append(f"  - {issue}")

    return [
        {
            "content_type": "text",
            "content_text": "\n".join(lines),
            "content_url": "",
            "parameter": {
                "checkpoint": "composition_result",
                "scene_path": scene_path,
                "imported_count": len(imported),
                "failed_count": len(failed),
                "review": review,
            },
        }
    ]
