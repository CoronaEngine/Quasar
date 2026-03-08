"""
集成对话：多场景室内设计工作流（LangGraph DAG 重构版）

将原单节点耦合流程拆分为 5 个独立节点的 DAG：
  analyzer_node → human_review_node
      → generate_images_node      ─┐
      → generate_layout_text_node ─┤→ aggregate_result_node → END

支持：多模态输入、Human-in-the-loop 审核、并行图文生成。
保持对外接口兼容（function_id、WORKFLOWS 导出、output_llm_content 结构）。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import base64
from pathlib import Path
import re
import time
from typing import Any, Dict, List, TYPE_CHECKING

import httpx

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ai_config.ai_config import get_ai_config
from ai_models.base_pool import get_chat_model, get_pool_registry, MediaCategory, OmniRequest
from ai_tools.registry import get_tool_registry
from ai_workflow.state import WorkflowState
from ai_tools.response_adapter import FILEID_SCHEME

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

MULTI_SCENE_FUNCTION_ID = 21001

# ---------------------------------------------------------------------------
# 工具 / 辅助函数（复用原有逻辑）
# ---------------------------------------------------------------------------


def _get_generate_image_tool():
    """惰性加载图片生成工具"""
    registry = get_tool_registry()
    if not registry.list_tools():
        from ai_tools.load_tools import load_tools

        load_tools(get_ai_config())
    return {t.name: t for t in registry.list_tools()}.get("generate_image")


def _extract_text(response: Any) -> str:
    """从 LLM response 中提取纯文本"""
    content = getattr(response, "content", "")
    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_blocks.append(str(block.get("text", "")))
        return "\n".join(text_blocks)
    return str(content or "")


def _extract_image_url(raw_result: Any) -> str:
    """从工具返回值中提取并解析图片 URL（含 fileid 延迟解析）。"""
    try:
        parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        part = parsed["llm_content"][0]["part"][0]
        extracted = str(part.get("content_url") or part.get("content_text") or "")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        extracted = str(raw_result)

    if extracted.count("{") > 1:
        return ""

    if extracted.startswith(FILEID_SCHEME):
        from ai_media_resource import get_media_registry

        file_id = extracted[len(FILEID_SCHEME):]
        try:
            # 阻塞等待异步任务完成，获取可访问 URL。
            return get_media_registry().resolve(file_id)
        except Exception as e:
            logger.error(f"[Workflow][generate_images] file_id 解析失败: {file_id}, err={e}")
            return ""

    return extracted


def _to_display_url(url: str) -> str:
    """将本地绝对路径转换为 file:// URL，便于 markdown 展示。"""
    if not url:
        return ""
    lowered = url.lower()
    if lowered.startswith(("http://", "https://", "data:", "file://")):
        return url

    path = Path(url)
    if path.is_absolute():
        return path.as_uri()
    return url


def _build_llm_content(text_parts: List[str]) -> List[Dict[str, Any]]:
    """构建兼容旧系统的 output_llm_content 列表"""
    entries: List[Dict[str, Any]] = []
    timestamp = int(time.time())
    for text in text_parts:
        entries.append(
            {
                "role": "assistant",
                "interface_type": "integrated",
                "sent_time_stamp": timestamp,
                "part": [
                    {
                        "content_type": "text",
                        "content_text": text,
                        "content_url": "",
                        "parameter": {},
                    }
                ],
            }
        )
    return entries


def _clean_json_text(raw: str) -> str:
    """去除 markdown code block 包裹，提取纯 JSON 文本"""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def _get_llm(temperature: float = 0.6):
    """获取聊天模型的便捷封装"""
    cfg = get_ai_config()
    chat_cfg = cfg.chat
    return get_chat_model(
        provider_name=chat_cfg.provider,
        model_name=chat_cfg.model,
        temperature=temperature,
        request_timeout=chat_cfg.request_timeout,
    )


# ---------------------------------------------------------------------------
# 回退元素（解析失败时兜底，避免后续节点空崩溃）
# ---------------------------------------------------------------------------

_FALLBACK_ELEMENTS: List[Dict[str, str]] = [
    {
        "item_name": "现代沙发",
        "image_prompt": (
            "A modern minimalist sofa, clean design, isolated on pure white "
            "background, studio lighting, octane render, masterpiece"
        ),
        "layout_desc": "放置于客厅中央，搭配浅色地毯与茶几。",
    },
    {
        "item_name": "艺术落地灯",
        "image_prompt": (
            "An artistic floor lamp, contemporary design, isolated on white "
            "background, soft studio lighting, product photography, masterpiece"
        ),
        "layout_desc": "置于沙发侧旁，提供柔和氛围照明。",
    },
    {
        "item_name": "装饰画",
        "image_prompt": (
            "A framed abstract wall art, modern style, isolated on pure white "
            "background, studio lighting, high quality render"
        ),
        "layout_desc": "悬挂于沙发上方墙面，作为空间视觉焦点。",
    },
]

# ---------------------------------------------------------------------------
# Node 1: analyzer_node — 结构化方案抽取
# ---------------------------------------------------------------------------

_ANALYZER_SYSTEM_PROMPT = """\
你是资深室内设计师兼 AI 助手。请根据用户提供的设计需求，构思 3-5 个核心设计单品/元素，
并为每个单品提供：
1. item_name —— 中文名称（简洁明了）
2. image_prompt —— 英文 AI 绘画 Prompt（包含物品描述、风格、纯白背景、产品摄影、\
高质量渲染等关键词）
3. layout_desc —— 该物品在空间中的布局与搭配建议（中文，1-2 句即可）

请 **严格** 以如下 JSON 数组格式输出（不要输出任何多余文本）：
[
  {
    "item_name": "物品名称",
    "image_prompt": "A modern minimalist sofa, clean lines, isolated on pure white \
background, studio lighting, octane render, masterpiece",
    "layout_desc": "放置于客厅中央，搭配浅色地毯与茶几形成会客区。"
  }
]
"""

_ANALYZER_MULTIMODAL_SUFFIX = (
    "\n\n【参考图片视觉分析】\n"
    "以下是 VLM 对用户提供的参考图片的分析结果，请结合此信息提取设计元素：\n"
)

_VLM_ANALYSIS_PROMPT = (
    "你是室内设计领域的视觉分析专家。请仔细观察图片，描述其中的：\n"
    "1. 主要家具与装饰物品（名称、材质、颜色、风格）\n"
    "2. 空间布局特点（动线、功能分区）\n"
    "3. 整体设计风格与氛围\n"
    "请用结构化的中文描述，便于后续提取设计元素。"
)


def _analyze_images_with_vlm(
    images: List[str], session_id: str = ""
) -> str:
    """通过项目 Omni 模块调用 VLM 对图片进行视觉分析。

    Returns:
        VLM 返回的文本分析结果；调用失败时返回空字符串。
    """
    try:
        from ai_media_resource import get_media_registry

        normalized_images: List[str] = []
        src_stats = {"fileid": 0, "file": 0, "local": 0, "http": 0, "data": 0, "other": 0}

        for raw in images:
            u = str(raw or "").strip()
            if not u:
                continue

            if u.startswith("data:"):
                src_stats["data"] += 1
                normalized_images.append(u)
                continue

            if u.startswith(FILEID_SCHEME):
                src_stats["fileid"] += 1
                u = str(get_media_registry().resolve(u[len(FILEID_SCHEME):]))

            if u.startswith("file://"):
                src_stats["file"] += 1
                from ai_models.utils import file_url_to_data_uri

                normalized_images.append(file_url_to_data_uri(u))
                continue

            p = Path(u)
            if p.exists():
                src_stats["local"] += 1
                from ai_models.utils import file_url_to_data_uri

                normalized_images.append(file_url_to_data_uri(p.resolve().as_uri()))
                continue

            if u.startswith(("http://", "https://")):
                src_stats["http"] += 1
                # 兜底：下载后转 data URI，避免上游 VLM 拉取远端 URL 失败。
                with httpx.Client(timeout=30.0, follow_redirects=True) as c:
                    r = c.get(u)
                    r.raise_for_status()
                    mime = r.headers.get("content-type", "").split(";")[0].strip().lower()
                    if not mime.startswith("image/"):
                        raise ValueError(f"VLM 输入不是图片: {u[:120]}, content-type={mime}")
                    b64 = base64.b64encode(r.content).decode("utf-8")
                    normalized_images.append(f"data:{mime};base64,{b64}")
                continue

            src_stats["other"] += 1
            logger.warning(f"[Workflow][analyzer] 无法识别的图片输入，已跳过: {u[:160]}")

        logger.info(
            "[Workflow][analyzer] image source stats: fileid=%s file=%s local=%s http=%s data=%s other=%s",
            src_stats["fileid"],
            src_stats["file"],
            src_stats["local"],
            src_stats["http"],
            src_stats["data"],
            src_stats["other"],
        )

        if not normalized_images:
            logger.warning("[Workflow][analyzer] 无可用图片传给 VLM")
            return ""

        pool_registry = get_pool_registry()
        request = OmniRequest(
            session_id=session_id or f"workflow-{int(time.time())}",
            prompt=_VLM_ANALYSIS_PROMPT,
            image_urls=normalized_images,
        )
        task = pool_registry.create_task(MediaCategory.OMNI, request)
        if task is None:
            logger.warning("[Workflow][analyzer] Omni 池无可用账号，跳过视觉分析")
            return ""
        result = task()  # 同步阻塞
        analysis = result.metadata.get("analysis_result", "")
        if analysis:
            logger.info(f"[Workflow][analyzer] VLM 分析完成，结果长度 {len(analysis)}")
        return analysis
    except Exception as e:
        logger.warning(f"[Workflow][analyzer] VLM 视觉分析失败: {e}")
        return ""


def analyzer_node(state: WorkflowState) -> Dict[str, Any]:
    """分析用户需求，提取结构化设计元素列表（extracted_elements）。

    多模态输入时通过 Omni 模块（VLM）先对图片做视觉分析，
    再将分析结果与用户文本需求一起传入文本 LLM 提取结构化元素。
    VLM 不可用时降级为纯文本分析。
    """
    if state.get("error"):
        return {}

    user_input = (state.get("prompt") or "").strip()
    if not user_input:
        return {"error": "缺少设计需求文本"}

    images = state.get("images") or []
    is_multimodal = bool(images)
    session_id = state.get("session_id", "")

    try:
        llm = _get_llm(temperature=0.6)

        # 有图片时通过 Omni VLM 做视觉分析，再将结果拼入文本提示
        system_text = _ANALYZER_SYSTEM_PROMPT
        user_content = f"用户需求：{user_input}"

        if is_multimodal:
            vlm_analysis = _analyze_images_with_vlm(images, session_id)
            if vlm_analysis:
                system_text += _ANALYZER_MULTIMODAL_SUFFIX
                user_content += f"\n\n{vlm_analysis}"
            else:
                logger.info("[Workflow][analyzer] VLM 分析无结果，仅用文本分析")

        response = llm.invoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=user_content),
            ]
        )
        raw_text = _extract_text(response)

        # --- 健壮 JSON 解析 ---
        cleaned = _clean_json_text(raw_text)
        parsed = json.loads(cleaned)

        # 兼容对象包裹数组的情况 {"elements": [...]}
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

        if not isinstance(parsed, list) or len(parsed) == 0:
            raise ValueError("解析结果不是非空数组")

        # 字段校验与补全
        elements: List[Dict[str, str]] = []
        for item in parsed:
            elements.append(
                {
                    "item_name": str(item.get("item_name", "未命名单品")),
                    "image_prompt": str(item.get("image_prompt", "")),
                    "layout_desc": str(item.get("layout_desc", "")),
                }
            )

        logger.info(f"[Workflow][analyzer] 提取到 {len(elements)} 个设计元素")
        return {
            "is_multimodal": is_multimodal,
            "extracted_elements": elements,
        }

    except json.JSONDecodeError as e:
        logger.warning(f"[Workflow][analyzer] JSON 解析失败，使用回退元素: {e}")
        return {
            "is_multimodal": is_multimodal,
            "extracted_elements": list(_FALLBACK_ELEMENTS),
        }
    except Exception as e:
        logger.error(f"[Workflow][analyzer] 执行异常: {e}")
        return {"error": f"方案分析失败: {e}"}


# ---------------------------------------------------------------------------
# Node 2: human_review_node — 人机审核中断点
# ---------------------------------------------------------------------------


def human_review_node(state: WorkflowState) -> Dict[str, Any]:
    """Human-in-the-loop 审核节点。

    优先使用 LangGraph ``interrupt`` 暂停图执行，将待审核元素返回给调用方。
    调用方 resume 时传入审核后的元素列表。
    若运行环境不支持 interrupt（无 checkpointer / 非交互调用），回退为自动通过，
    并在 intermediate["human_review"] 中标记此回退点，便于后续接 UI 人审。
    """
    if state.get("error"):
        return {}

    extracted = state.get("extracted_elements", [])
    if not extracted:
        return {"error": "无可审核的设计元素"}

    # --- 尝试 LangGraph interrupt（真正的 HITL）---
    try:
        from langgraph.types import interrupt  # type: ignore[import-untyped]

        review_payload = {
            "action": "review_elements",
            "elements": extracted,
            "message": "请审核以下设计元素，可修改后返回，或原样返回表示通过。",
        }
        approved = interrupt(review_payload)

        if isinstance(approved, list) and len(approved) > 0:
            logger.info(
                f"[Workflow][human_review] 人工审核通过，{len(approved)} 个元素"
            )
            return {"approved_elements": approved}
    except Exception as exc:
        logger.info(
            f"[Workflow][human_review] interrupt 不可用或被跳过 ({exc})，自动通过"
        )

    # --- 回退：自动通过 ---
    logger.info(
        f"[Workflow][human_review] 自动通过 {len(extracted)} 个元素（回退模式）"
    )
    return {
        "approved_elements": extracted,
        "intermediate": {
            **state.get("intermediate", {}),
            "human_review": {
                "status": "auto_approved",
                "elements": extracted,
                "note": "当前为自动通过回退，后续可接入 UI 实现真正的人机审核。",
            },
        },
    }


# ---------------------------------------------------------------------------
# Node 3A: generate_images_node — 并发图片生成
# ---------------------------------------------------------------------------


def generate_images_node(state: WorkflowState) -> Dict[str, Any]:
    """并发生成所有审核通过元素的图片。

    使用 ThreadPoolExecutor 并行调用图片生成工具。
    单项失败仅跳过该项并记录日志，不抛出致命异常。
    """
    if state.get("error"):
        return {}

    approved = state.get("approved_elements", [])
    if not approved:
        logger.warning("[Workflow][generate_images] 无审核通过的元素")
        return {"generated_images": {}}

    image_tool = _get_generate_image_tool()
    if not image_tool:
        logger.warning("[Workflow][generate_images] 图片生成工具不可用")
        return {"generated_images": {}}

    generated: Dict[str, str] = {}

    def _generate_one(element: Dict[str, str]) -> tuple:
        name = element.get("item_name", "未命名")
        prompt = element.get("image_prompt", "")
        if not prompt:
            return name, ""
        try:
            raw_result = image_tool.invoke({"prompt": prompt})
            image_url = _extract_image_url(raw_result)
            return name, image_url
        except Exception as e:
            logger.error(f"[Workflow][generate_images] {name} 生成失败: {e}")
            return name, ""

    max_workers = min(len(approved), 5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_generate_one, elem) for elem in approved]
        for future in concurrent.futures.as_completed(futures):
            try:
                name, url = future.result()
                if url:
                    generated[name] = url
            except Exception as e:
                logger.error(f"[Workflow][generate_images] 并发任务异常: {e}")

    logger.info(
        f"[Workflow][generate_images] 成功生成 "
        f"{len(generated)}/{len(approved)} 张图片"
    )
    return {"generated_images": generated}


# ---------------------------------------------------------------------------
# Node 3B: generate_layout_text_node — 排版（纯格式化，不调用 LLM）
# ---------------------------------------------------------------------------


def generate_layout_text_node(state: WorkflowState) -> Dict[str, Any]:
    """将审核通过的元素格式化为物品清单与布局描述文本。

    不调用 LLM，仅对已有的 item_name / layout_desc 做 Markdown 排版。
    """
    if state.get("error"):
        return {}

    approved = state.get("approved_elements", [])
    if not approved:
        return {"layout_text": "暂无设计元素。"}

    lines: List[str] = ["## 设计方案\n"]
    for idx, e in enumerate(approved, 1):
        name = e.get("item_name", "未命名")
        desc = e.get("layout_desc", "")
        lines.append(f"### {idx}. {name}")
        if desc:
            lines.append(f"{desc}\n")

    layout_text = "\n".join(lines)
    logger.info("[Workflow][generate_layout_text] 排版完成")
    return {"layout_text": layout_text}


# ---------------------------------------------------------------------------
# Node 4: aggregate_result_node — 结果聚合
# ---------------------------------------------------------------------------


def aggregate_result_node(state: WorkflowState) -> Dict[str, Any]:
    """汇总物品清单、布局描述与生成图片，输出 Markdown 格式的 output_llm_content。"""
    generated_images: Dict[str, str] = state.get("generated_images", {})
    approved = state.get("approved_elements", [])

    # 按物品逐项输出：名称 + 布局描述 + 图片
    md_parts: List[str] = ["## 设计方案\n"]

    for idx, e in enumerate(approved, 1):
        name = e.get("item_name", "未命名")
        desc = e.get("layout_desc", "")
        md_parts.append(f"### {idx}. {name}")
        if desc:
            md_parts.append(f"{desc}")
        img_url = generated_images.get(name, "")
        if img_url:
            md_parts.append(f"\n![{name}]({_to_display_url(img_url)})\n")
        else:
            md_parts.append("\n（图片生成失败）\n")

    final_markdown = "\n".join(md_parts)
    output_content = _build_llm_content([final_markdown])

    intermediate = {
        **state.get("intermediate", {}),
        "workflow": "integrated_multi_scene",
        "element_count": len(approved),
        "image_success_count": len(generated_images),
    }

    logger.info(
        f"[Workflow][aggregate] 完成：{len(approved)} 个元素，"
        f"{len(generated_images)} 张图片成功"
    )

    return {
        "output_llm_content": output_content,
        "intermediate": intermediate,
    }


# ---------------------------------------------------------------------------
# DAG 构建与导出
# ---------------------------------------------------------------------------


def build_multi_scene_workflow() -> "CompiledStateGraph":
    """构建多场景室内设计 LangGraph DAG。

    拓扑：
        START → analyzer → human_review ─→ generate_images       ─┐
                                         └→ generate_layout_text ─┤→ aggregate_result → END
    """
    graph = StateGraph(WorkflowState)

    # 注册节点
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("generate_images", generate_images_node)
    graph.add_node("generate_layout_text", generate_layout_text_node)
    graph.add_node("aggregate_result", aggregate_result_node)

    # 编排边：线性段
    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "human_review")

    # 并行分支：人审后同时启动图片生成与文案生成
    graph.add_edge("human_review", "generate_images")
    graph.add_edge("human_review", "generate_layout_text")

    # 汇聚：两个并行分支完成后进入聚合节点
    graph.add_edge("generate_images", "aggregate_result")
    graph.add_edge("generate_layout_text", "aggregate_result")

    graph.add_edge("aggregate_result", END)

    return graph.compile()


WORKFLOWS: Dict[int, "CompiledStateGraph"] = {
    MULTI_SCENE_FUNCTION_ID: build_multi_scene_workflow(),
}

__all__ = ["WORKFLOWS", "MULTI_SCENE_FUNCTION_ID", "build_multi_scene_workflow"]
