from __future__ import annotations

from typing import Any, Dict, List

from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import FormatterFunc

NO_OUTPUT: FormatterFunc = lambda _data, _state: []


def _count_stats(model_results: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计检索/生成/失败数量。"""
    return {
        "total": len(model_results),
        "retrieval_count": sum(
            1 for row in model_results if row.get("source") == "retrieval"
        ),
        "generation_count": sum(
            1
            for row in model_results
            if row.get("source") == "generation" and not row.get("error")
        ),
        "error_count": sum(1 for row in model_results if row.get("error")),
    }


def _build_user_visible_result(
    model_results: List[Dict[str, Any]],
    title: str,
    summary_prefix: str,
    include_register_status: bool,
    stats_override: Dict[str, int] | None = None,
) -> tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    """构建统一的用户可视化结果文本与精简结构化条目。"""
    stats = dict(stats_override or _count_stats(model_results))

    lines: List[str] = [
        title,
        (
            f"{summary_prefix} **{len(model_results)}** 个物体："
            f"检索命中 **{stats.get('retrieval_count', 0)}**，"
            f"新生成 **{stats.get('generation_count', 0)}**，"
            f"失败 **{stats.get('error_count', 0)}**"
        ),
        "",
    ]

    items: List[Dict[str, Any]] = []
    for row in model_results:
        name = row.get("item_name", "未知")
        source = row.get("source", "")
        error = str(row.get("error", "") or "").strip()

        if source == "retrieval" and not error:
            object_id = row.get("object_id", "")
            distance = row.get("distance", 0)
            lines.append(f"- {name}: 复用已有模型（ID: {object_id}, 距离: {distance:.4f}）")
            items.append(
                {
                    "item_name": name,
                    "status": "retrieval",
                    "object_id": object_id,
                    "distance": distance,
                }
            )
            continue

        if source == "generation" and not error:
            model_path = row.get("model_path", "")
            register_status = row.get("register_status", "")
            register_text = (
                f"，入库: {register_status}"
                if include_register_status and register_status
                else ""
            )
            lines.append(f"- {name}: 已生成新模型（{model_path}{register_text}）")
            item = {
                "item_name": name,
                "status": "generation",
                "model_path": model_path,
            }
            if include_register_status:
                item["register_status"] = register_status
            items.append(item)
            continue

        shown_error = error or "处理失败"
        lines.append(f"- {name}: 失败（{shown_error}）")
        items.append(
            {
                "item_name": name,
                "status": "error",
                "error": shown_error,
            }
        )

    return lines, items, stats


def format_retrieve_or_generate_checkpoint_parts(
    data: Dict[str, Any],
    _state: ModelRetrievalWorkflowState,
) -> List[Dict[str, Any]]:
    """为 retrieve_or_generate 检查点输出可视化摘要。"""
    model_results = data.get("model_results", [])
    if not isinstance(model_results, list) or not model_results:
        return []

    lines, preview_items, stats = _build_user_visible_result(
        model_results=model_results,
        title="## 模型检索阶段结果",
        summary_prefix="已处理",
        include_register_status=False,
    )

    return [
        {
            "content_type": "text",
            "content_text": "\n".join(lines),
            "content_url": "",
            "parameter": {
                "checkpoint": "retrieve_or_generate",
                "summary": stats,
                "items": preview_items,
            },
        }
    ]


def format_result_checkpoint_parts(
    data: Dict[str, Any],
    state: ModelRetrievalWorkflowState,
) -> List[Dict[str, Any]]:
    """为 format_result 检查点输出面向用户的最终可视化结果。"""
    model_results = state.get("model_results", [])
    mr_stats = data.get("global_assets", {}).get("model_retrieval", {})

    lines, result_items, stats = _build_user_visible_result(
        model_results=model_results,
        title="## 模型检索与 3D 生成结果",
        summary_prefix="总计",
        include_register_status=True,
        stats_override={
            "total": len(model_results),
            "retrieval_count": mr_stats.get("retrieval_count", 0),
            "generation_count": mr_stats.get("generation_count", 0),
            "error_count": mr_stats.get("error_count", 0),
        },
    )

    return [
        {
            "content_type": "text",
            "content_text": "\n".join(lines),
            "content_url": "",
            "parameter": {
                "checkpoint": "format_result",
                "summary": stats,
                "items": result_items,
            },
        }
    ]
