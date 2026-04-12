from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ai_workflow.progress import publish_node_entries_event
from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import FormatterFunc, build_node_dialogue_entry

from .helpers import find_sibling_preview_image

NO_OUTPUT: FormatterFunc = lambda _data, _state: []


def _to_display_url(url: str) -> str:
    """将本地绝对路径转换为可展示 URL。"""
    text = str(url or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "data:", "file://")):
        return text

    path = Path(text)
    if path.is_absolute():
        return path.as_uri()
    return text


def _is_displayable_url(url: str) -> bool:
    """判断路径是否可直接提供给前端展示。"""
    text = str(url or "").strip()
    if not text:
        return False

    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "data:", "file://")):
        return True

    return Path(text).is_absolute()


def _normalize_url_key(url: str) -> str:
    """归一化 URL/路径，便于做等价比较。"""
    text = str(url or "").strip()
    if not text:
        return ""

    if text.lower().startswith("file://"):
        text = text[7:]

    return text.replace("\\", "/").strip().lower()


def _collect_list_field(container: Dict[str, Any], key: str, out: List[str]) -> None:
    """将 container[key] (str 或 list) 中的元素追加到 out。"""
    value = container.get(key)
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        out.extend(str(item or "") for item in value)


def _pick_displayable_preview_path(row: Dict[str, Any]) -> str:
    """优先从候选预览图中选择真正可显示的一条。"""
    candidates: List[str] = []

    _collect_list_field(row, "preview_paths", candidates)
    _collect_list_field(row, "image_paths", candidates)

    parameter = row.get("parameter", {})
    if isinstance(parameter, dict):
        _collect_list_field(parameter, "preview_paths", candidates)

    input_key = _normalize_url_key(str(row.get("input_image_url", "") or ""))

    # 第一优先级：可展示且与输入图不同，避免误把输入图当作生成预览图。
    for candidate in candidates:
        if not _is_displayable_url(candidate):
            continue
        if input_key and _normalize_url_key(candidate) == input_key:
            continue
        return str(candidate).strip()

    for candidate in candidates:
        if _is_displayable_url(candidate):
            return str(candidate).strip()

    return ""


def _make_text_part(content_text: str) -> Dict[str, Any]:
    return {
        "content_type": "text",
        "content_text": content_text,
        "content_url": "",
        "parameter": {},
    }


def _make_image_part(content_url: str, description: str = "") -> Dict[str, Any]:
    return {
        "content_type": "image",
        "content_text": description,
        "content_url": _to_display_url(content_url),
        "parameter": {},
    }


def _select_preview_for_row(row: Dict[str, Any]) -> str:
    """按分支规则为一条结果选择前端展示图。"""
    preview_path = _pick_displayable_preview_path(row)
    if preview_path:
        return preview_path

    if str(row.get("source", "") or "") == "retrieval":
        model_dir_image = find_sibling_preview_image(str(row.get("model_path", "") or ""))
        if model_dir_image and _is_displayable_url(model_dir_image):
            return model_dir_image

    input_image_url = str(row.get("input_image_url", "") or "")
    if input_image_url and _is_displayable_url(input_image_url):
        return input_image_url

    return ""


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


def _append_preview_part(parts: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    preview_image_url = _select_preview_for_row(row)
    if preview_image_url:
        parts.append(
            _make_image_part(
                preview_image_url,
                f"模型预览 - {row.get('item_name', '未知')}",
            )
        )


def _format_progress_parts(
    row: Dict[str, Any],
    *,
    title: str,
    detail: str,
    done_count: int,
    total_count: int,
) -> List[Dict[str, Any]]:
    summary = f"## {title}\n已完成 **{done_count}/{total_count}** 项\n{detail}"
    parts: List[Dict[str, Any]] = [_make_text_part(summary)]
    _append_preview_part(parts, row)
    return parts


def _build_user_visible_parts(
    model_results: List[Dict[str, Any]],
    title: str,
    summary_prefix: str,
    include_register_status: bool,
    stats_override: Dict[str, int] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """构建统一的用户可视化结果 parts。"""
    stats = dict(stats_override or _count_stats(model_results))

    parts: List[Dict[str, Any]] = [
        _make_text_part(
            "\n".join(
                [
                    title,
                    (
                        f"{summary_prefix} **{len(model_results)}** 个物体："
                        f"检索命中 **{stats.get('retrieval_count', 0)}**，"
                        f"新生成 **{stats.get('generation_count', 0)}**，"
                        f"失败 **{stats.get('error_count', 0)}**"
                    ),
                ]
            )
        )
    ]

    for row in model_results:
        name = row.get("item_name", "未知")
        source = row.get("source", "")
        error = str(row.get("error", "") or "").strip()

        if source == "retrieval" and not error:
            object_id = row.get("object_id", "")
            distance = row.get("distance", 0)
            parts.append(
                _make_text_part(
                    f"- {name}: 复用已有模型（ID: {object_id}, 距离: {distance:.4f}）"
                )
            )
            _append_preview_part(parts, row)
            continue

        if source == "generation" and not error:
            model_path = row.get("model_path", "")
            register_status = row.get("register_status", "")
            register_text = (
                f"，入库: {register_status}"
                if include_register_status and register_status
                else ""
            )
            parts.append(
                _make_text_part(f"- {name}: 已生成新模型（{model_path}{register_text}）")
            )
            _append_preview_part(parts, row)
            continue

        shown_error = error or "处理失败"
        parts.append(_make_text_part(f"- {name}: 失败（{shown_error}）"))
        _append_preview_part(parts, row)

    return parts, stats


def _build_checkpoint_items(model_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构建供前端 checkpoint 面板消费的精简结果。"""
    items: List[Dict[str, Any]] = []
    for row in model_results:
        name = row.get("item_name", "未知")
        source = row.get("source", "")
        error = str(row.get("error", "") or "").strip()

        if source == "retrieval" and not error:
            items.append(
                {
                    "item_name": name,
                    "status": "retrieval",
                    "object_id": row.get("object_id", ""),
                    "distance": row.get("distance", 0),
                }
            )
            continue

        if source == "generation" and not error:
            item = {
                "item_name": name,
                "status": "generation",
                "model_path": row.get("model_path", ""),
            }
            if row.get("register_status"):
                item["register_status"] = row.get("register_status")
            if row.get("retry_count"):
                item["retry_count"] = row.get("retry_count")
            items.append(item)
            continue

        items.append(
            {
                "item_name": name,
                "status": "error",
                "error": error or "处理失败",
            }
        )

    return items


def _detail_for_retrieve(row: Dict[str, Any], name: str) -> str:
    error = str(row.get("search_error", "") or row.get("error", "") or "").strip()
    if str(row.get("source", "") or "") == "retrieval" and not error:
        return f"- {name}: 命中模型（ID: {row.get('object_id', '')}, 距离: {row.get('distance', 0):.4f}）"
    if error:
        return f"- {name}: 检索失败，转入生成（{error}）"
    return f"- {name}: 未命中，转入生成"


def _detail_for_generate(row: Dict[str, Any], name: str) -> str:
    error = str(row.get("error", "") or "").strip()
    if error:
        return f"- {name}: 生成失败（{error}）"
    return f"- {name}: 已生成新模型（{row.get('model_path', '')}）"


def _detail_for_register(row: Dict[str, Any], name: str) -> str:
    status = str(row.get("register_status", "skipped") or "skipped").lower()
    if status == "inserted":
        return f"- {name}: 新增入库成功"
    if status == "updated":
        return f"- {name}: 更新入库成功"
    if status == "failed":
        error = str(row.get("register_error", "") or row.get("error", "入库失败")).strip()
        return f"- {name}: 入库失败（{error or '未知错误'}）"
    return f"- {name}: 跳过入库"


_PROGRESS_FORMATTERS: Dict[str, tuple[str, Any]] = {
    "retrieve": ("模型检索进度", _detail_for_retrieve),
    "generate": ("3D 生成进度", _detail_for_generate),
    "register": ("模型入库进度", _detail_for_register),
}


def publish_node_progress(
    state: ModelRetrievalWorkflowState,
    result: Dict[str, Any],
    *,
    node_name: str,
    done_count: int,
    total_count: int,
) -> None:
    """统一发布单个任务的流式进度输出。"""
    title, detail_fn = _PROGRESS_FORMATTERS[node_name]
    name = result.get("item_name", "未知")
    parts = _format_progress_parts(
        result,
        title=title,
        detail=detail_fn(result, name),
        done_count=done_count,
        total_count=total_count,
    )
    if not parts:
        return

    entry = build_node_dialogue_entry(
        "integrated",
        parts,
        node_name=node_name,
        function_id=state.get("function_id"),
    )
    publish_node_entries_event(
        str(state.get("session_id", "default") or "default"),
        node_name,
        [entry],
    )


def format_result_checkpoint_parts(
    data: Dict[str, Any],
    state: ModelRetrievalWorkflowState,
) -> List[Dict[str, Any]]:
    """为 format_result 检查点输出面向用户的最终可视化结果。"""
    model_results = state.get("model_results", [])
    mr_stats = data.get("global_assets", {}).get("model_retrieval", {})
    stats = {
        "total": len(model_results),
        "retrieval_count": mr_stats.get("retrieval_count", 0),
        "generation_count": mr_stats.get("generation_count", 0),
        "error_count": mr_stats.get("error_count", 0),
    }

    parts, _stats = _build_user_visible_parts(
        model_results=model_results,
        title="## 模型检索与 3D 生成结果",
        summary_prefix="总计",
        include_register_status=True,
        stats_override=stats,
    )

    if parts:
        parts[0]["parameter"] = {
            "checkpoint": "format_result",
            "summary": stats,
            "items": _build_checkpoint_items(model_results),
        }

    return parts
