from __future__ import annotations

import logging
import queue

from typing import Any, Dict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ...ai_agent.executor import stream_agent
from ...ai_tools.common import build_error_response

from .common import (
    ToolError,
    build_heartbeat_signal,
    build_stream_done_signal,
    build_success_chunk,
    build_tool_error_entry,
    extract_text_parts,
    is_recoverable_tool_error,
    make_assistant_entry,
    resolve_tool_message,
)
from .context import prepare_stream_context
from .stream_utils import run_in_thread, with_heartbeat
from .workflow_bridge import resolve_and_stream_workflow

logger = logging.getLogger(__name__)


def _summarize_global_assets(assets: Dict[str, Any]) -> str:
    """将 global_assets 摘要为 Agent 可读的文本。"""
    lines = ["[工作流上下文] 以下是之前工作流产生的资产信息，你可以引用："]

    # 模型检索结果
    mr = assets.get("model_retrieval", {})
    model_results = mr.get("model_results", [])
    if model_results:
        lines.append(f"\n## 可用 3D 模型 ({len(model_results)} 个):")
        for row in model_results:
            name = row.get("item_name", "未知")
            path = row.get("model_path", "")
            source = row.get("source", "")
            error = row.get("error", "")
            if error:
                lines.append(f"  - {name}: 失败 ({error})")
            else:
                lines.append(f"  - {name}: {path} (来源: {source})")

    # 场景组合结果
    sc = assets.get("scene_composition", {})
    if sc:
        scene_path = sc.get("scene_path", "")
        imported = sc.get("imported_count", 0)
        review = sc.get("review_result", {})
        lines.append(f"\n## 场景组合结果:")
        lines.append(f"  - 场景文件: {scene_path}")
        lines.append(f"  - 已导入模型: {imported}")
        if review:
            lines.append(f"  - 审查: {review.get('overall', 'N/A')} (评分: {review.get('score', 'N/A')})")

    # 多场景设计
    ms = assets.get("multi_scene", {})
    approved = ms.get("approved_elements", [])
    if approved:
        names = [e.get("name", "?") for e in approved[:10]]
        lines.append(f"\n## 设计元素 ({len(approved)} 个): {', '.join(names)}")

    return "\n".join(lines)


def _inject_global_assets_context(
    session_id: str,
    pending_history: list,
) -> list:
    """若 session 存在 global_assets，在 history 开头注入摘要 SystemMessage。"""
    try:
        from ...ai_workflow.loop_state import get_loop_global_assets

        assets = get_loop_global_assets(session_id)
        if not assets:
            return pending_history

        summary = _summarize_global_assets(assets)
        if not summary or summary.count("\n") <= 1:
            return pending_history

        return [SystemMessage(content=summary)] + list(pending_history)
    except Exception:
        logger.debug("注入 global_assets 上下文失败，跳过", exc_info=True)
        return pending_history


def handle_integrated_entrance_stream_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
):
    """流式统一聊天接口内部实现。"""
    try:
        logger.info("收到 integrated stream 请求")
        logger.debug(f"请求详情: {request_data}")
        llm_content = request_data.get("llm_content", [])
        if not isinstance(llm_content, list) or not llm_content:
            raise ValueError("llm_content 不能为空")

        workflow_stream = resolve_and_stream_workflow(
            request_data,
            interface_type="integrated",
        )
        if workflow_stream is not None:
            logger.info("integrated stream 请求命中 workflow 路由（流式）")
            yield from workflow_stream
            yield build_stream_done_signal(session_id, metadata)
            return

        from ...ai_agent.protocol import extract_tool_media_parts
        from ...ai_agent.conversation import update_history

        ctx = prepare_stream_context(request_data)
        session_id = ctx["session_id"]
        pending_history = ctx["pending_history"]
        media_registry = ctx["media_registry"]

        # 将工作流循环中积累的 global_assets 注入到 Agent 上下文
        pending_history = _inject_global_assets_context(session_id, pending_history)

        accumulated_messages = list(pending_history)

        for chunk in with_heartbeat(stream_agent(pending_history), interval=5.0):
            if chunk is None:
                yield build_heartbeat_signal(session_id, metadata)
                continue

            for _, node_data in chunk.items():
                new_messages = node_data.get("messages", [])
                if not new_messages:
                    continue

                accumulated_messages.extend(new_messages)

                for msg in new_messages:
                    if isinstance(msg, AIMessage):
                        text_parts = extract_text_parts(msg)
                        if text_parts:
                            logger.debug(f"[Stream] yield AIMessage delta: {text_parts}")
                            yield build_success_chunk(
                                session_id, metadata, make_assistant_entry(text_parts)
                            )

                    elif isinstance(msg, ToolMessage):
                        result_queue = run_in_thread(resolve_tool_message, msg, "[Stream] ")
                        tool_parts = None
                        resolve_error = None
                        while True:
                            try:
                                status, value = result_queue.get(timeout=5.0)
                            except queue.Empty:
                                yield build_heartbeat_signal(session_id, metadata)
                                continue
                            if status == "error":
                                resolve_error = value
                            else:
                                tool_parts = value
                            break

                        if resolve_error is not None:
                            if isinstance(resolve_error, ToolError) and is_recoverable_tool_error(resolve_error):
                                error_entry = build_tool_error_entry(resolve_error)
                                yield build_success_chunk(session_id, metadata, error_entry)
                                accumulated_messages.append(
                                    AIMessage(content=error_entry["part"][0]["content_text"])
                                )
                                continue
                            raise resolve_error

                        if tool_parts is None:
                            error_entry = build_tool_error_entry(
                                ToolError("工具服务暂时不可用，请稍后重试。")
                            )
                            yield build_success_chunk(session_id, metadata, error_entry)
                            accumulated_messages.append(
                                AIMessage(content=error_entry["part"][0]["content_text"])
                            )
                            continue

                        if tool_parts:
                            logger.debug(f"[Stream] yield ToolMessage delta: {tool_parts}")
                            yield build_success_chunk(
                                session_id, metadata, make_assistant_entry(tool_parts)
                            )

        tool_media_parts = extract_tool_media_parts(accumulated_messages)
        if tool_media_parts:
            media_registry.register_batch(session_id, tool_media_parts)

        update_history(session_id, accumulated_messages)
        logger.info("integrated stream 请求处理完成")

        yield build_stream_done_signal(session_id, metadata)

    except Exception as exc:
        logger.error(f"处理 integrated stream 入口异常: {exc}", exc_info=True)
        error_response = build_error_response(
            interface_type="integrated",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )
        yield error_response


__all__ = ["handle_integrated_entrance_stream_inner"]
