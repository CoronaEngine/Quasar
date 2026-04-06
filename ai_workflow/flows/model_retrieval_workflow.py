"""
第二步工作流：模型检索与 3D 生成（LangGraph DAG）

接收第一步（多场景室内设计工作流）的输出状态，对每个物体：
  1. 使用 object_recognition 模块检索已有 3D 模型
  2. 若检索命中（distance < 阈值），记录模型 ID
  3. 若未命中，调用 three_d_generate 模块生成新 3D 模型

DAG 拓扑：
  START → dispatch_node → retrieve_or_generate_node → register_node → format_result_node → END

保持对外接口约定（function_id、WORKFLOWS / WORKFLOW_COMMANDS 导出）。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

from langgraph.graph import END, START, StateGraph

from ai_config.ai_config import get_ai_config
from ai_tools.registry import get_tool_registry
from ai_workflow.state import ModelRetrievalWorkflowState
from ai_workflow.streaming import FormatterFunc, stream_output_node
from config.app_config import get_app_config

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

MODEL_RETRIEVAL_FUNCTION_ID = 21002

# 检索距离阈值：低于此值视为命中
SEARCH_DISTANCE_THRESHOLD = 0.3
SEARCH_MAX_WORKERS = 1
GENERATION_MAX_WORKERS = 1

# ---------------------------------------------------------------------------
# 工作流内置测试样例
# ---------------------------------------------------------------------------

_TEST_CASE_DATA: Dict[str, Dict[str, Any]] = {
    "default": {
        # 模拟第一步工作流的 global_assets 结构
        "global_assets": {
            "multi_scene": {
                "approved_elements": [
                    {
                        "item_name": "现代沙发",
                        "image_prompt": "A modern minimalist sofa...",
                        "layout_desc": "放置于客厅中央",
                    },
                    {
                        "item_name": "艺术落地灯",
                        "image_prompt": "An artistic floor lamp...",
                        "layout_desc": "置于沙发侧旁",
                    },
                ],
                "generated_images": {
                    "现代沙发": "file://test_sofa.jpg",
                    "艺术落地灯": "file://test_lamp.jpg",
                },
            }
        },
        # 预期的模型检索/生成结果（可选，用于完整流程测试）
        "expected_model_results": [
            {
                "item_name": "现代沙发",
                "object_id": "modern_sofa",
                "source": "retrieval",
                "distance": 0.15,
            },
            {
                "item_name": "艺术落地灯",
                "object_id": "art_lamp",
                "source": "generation",
                "model_path": "/models/art_lamp/base.glb",
            },
        ],
    },
    "input_only": {
        # 仅提供第二步工作流所需输入，不提供任何预期输出。
        # dispatch 会使用这些输入组装任务，后续 retrieve / generate /
        # register / format_result 都走真实逻辑。
        "global_assets": {
            "multi_scene": {
                "approved_elements": [
                    {
                        "item_name": "现代沙发",
                        "image_prompt": "A modern minimalist sofa with clean lines, premium fabric, isolated on white background",
                        "layout_desc": "放置于客厅中央，形成主要会客区。",
                    },
                    {
                        "item_name": "艺术落地灯",
                        "image_prompt": "An artistic floor lamp, contemporary design, warm ambient style, isolated on white background",
                        "layout_desc": "置于沙发侧后方，提供辅助氛围照明。",
                    },
                ],
                "generated_images": {
                    "现代沙发": "D:\\CodeLib\\storage_root\\media_storage\\resource_4dafc270e3e44c7ea514215b406b80ab_c4efd90a-31a1-11f1-b3e9-68ecc582fadb.png",
                    "艺术落地灯": "D:\\CodeLib\\storage_root\\media_storage\\resource_4dafc270e3e44c7ea514215b406b80ab_c032ee4b-31a1-11f1-bbb2-68ecc582fadb.png",
                },
            }
        },
    },
}


def _get_test_case(test_case_key: str) -> Dict[str, Any]:
    """获取指定测试样例的完整 state 覆盖数据。

    Args:
        test_case_key: 样例键名，如 "default" 或 "input_only"

    Returns:
        该样例对应的 state 覆盖字段字典（包含 global_assets 和其他测试数据）。
        如果样例不存在，返回空字典。
    """
    case_data = _TEST_CASE_DATA.get(test_case_key or "default", {})
    logger.info(
        f"[ModelRetrieval][test_case] Loaded test case: {test_case_key or 'default'}, "
        f"fields={list(case_data.keys())}"
    )
    return dict(case_data)


def _normalize_object_id(name: str, fallback_index: int) -> str:
    """将物体名转换为 object_id 友好的目录名。"""
    cleaned = re.sub(r"\s+", "_", (name or "").strip())
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "_", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = f"object_{fallback_index:02d}"
    return cleaned[:64]


# ---------------------------------------------------------------------------
# 工具获取
# ---------------------------------------------------------------------------


def _get_tool(name: str) -> Any:
    """从工具注册表中按名称获取工具，按需触发懒加载。"""
    registry = get_tool_registry()
    tools = registry.list_tools()
    if not tools:
        from ai_tools.load_tools import load_tools

        load_tools(get_ai_config())
        tools = registry.list_tools()
    return {t.name: t for t in tools}.get(name)


def _get_search_tool():
    """获取物体搜索工具 (search_similar_object)"""
    return _get_tool("search_similar_object")


def _get_3d_generate_tool():
    """获取 3D 模型生成工具 (rodin_generate_3d)"""
    return _get_tool("rodin_generate_3d")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


# 不产生对话输出的节点使用空 formatter，仅借助装饰器做 error/awaiting_review 拦截。
_NO_OUTPUT: FormatterFunc = lambda _data, _state: []


def _parse_tool_result(raw_result: Any) -> Dict[str, Any]:
    """解析工具 envelope，统一返回字典结构。"""
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, str):
        return json.loads(raw_result)
    raise TypeError(f"不支持的工具返回类型: {type(raw_result)!r}")


def _extract_tool_error(parsed_result: Dict[str, Any]) -> str:
    """从工具 envelope 中提取错误信息。"""
    error_code = parsed_result.get("error_code", 0)
    if not error_code:
        return ""

    status_info = str(parsed_result.get("status_info", "") or "").strip()
    if status_info and status_info.lower() != "success":
        return status_info

    try:
        parts = parsed_result["llm_content"][0]["part"]
        for part in parts:
            text = str(part.get("content_text", "") or "").strip()
            if text:
                return text
    except (KeyError, IndexError, TypeError):
        pass

    return "工具调用失败"


def _parse_search_result(raw_result: Any) -> Dict[str, Any]:
    """解析 search_similar_object 返回值，提取 matches 与错误信息。"""
    try:
        parsed = _parse_tool_result(raw_result)
        error_message = _extract_tool_error(parsed)
        if error_message:
            return {"matches": [], "error": error_message}

        parts = parsed["llm_content"][0]["part"]
        for part in parts:
            matches = part.get("parameter", {}).get("matches", [])
            if isinstance(matches, list):
                return {"matches": matches, "error": ""}
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"matches": [], "error": "搜索结果解析失败"}


def _parse_3d_result(raw_result: Any) -> Dict[str, Any]:
    """解析 rodin_generate_3d 返回值，提取模型文件路径与元数据。

    工具返回的 envelope 中:
    - ``metadata`` 包含 ``model_folder``、``has_mesh_pending``、``object_id`` 等信息
    - ``parts`` 只包含 ``content_type == "image"`` 的预览图（mesh 在后台线程下载）
    - 模型文件路径通过 ``model_folder`` + 约定文件名 ``base.<ext>`` 推导
    """
    try:
        parsed = _parse_tool_result(raw_result)
        error_message = _extract_tool_error(parsed)
        if error_message:
            return {"error": error_message}

        metadata = parsed.get("metadata") or {}
        model_folder: str = metadata.get("model_folder", "")
        has_mesh_pending: bool = metadata.get("has_mesh_pending", False)
        meta_object_id: str = metadata.get("object_id", "")

        parts = parsed["llm_content"][0]["part"]
        preview_paths: List[str] = []
        geometry_file_format = "glb"

        for part in parts:
            if part.get("content_type") == "image":
                preview_path = part.get("content_text") or part.get("content_url") or ""
                if preview_path:
                    preview_paths.append(preview_path)
                # 尝试从 part parameter 中获取格式信息
                part_param = part.get("parameter") or {}
                fmt = part_param.get("geometry_file_format", "")
                if fmt:
                    geometry_file_format = fmt

        # metadata 中的格式优先
        geometry_file_format = metadata.get(
            "geometry_file_format", geometry_file_format
        )

        if model_folder:
            # 模型文件路径 = model_folder/base.<ext>，与后台下载的命名约定一致
            model_path = f"{model_folder}/base.{geometry_file_format}"
            parameter: Dict[str, Any] = {
                "preview_paths": preview_paths,
                "model_folder": model_folder,
                "geometry_file_format": geometry_file_format,
                "has_mesh_pending": has_mesh_pending,
                "object_id": meta_object_id,
            }
            return {
                "model_path": model_path,
                "parameter": parameter,
            }
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"error": "3D 生成结果解析失败"}


def _get_recognition_db_config() -> Dict[str, Any]:
    """读取 object_recognition 向量库配置，提供 register_node 使用。"""
    cfg = get_ai_config()
    raw = getattr(cfg, "object_recognition", None)

    db_path = str(get_app_config().paths.object_recognition_db)
    vector_dim = 1024

    if isinstance(raw, dict):
        vector_cfg = raw.get("vector_db", {}) or {}
        db_path = str(vector_cfg.get("db_path", db_path))
        vector_dim = int(vector_cfg.get("vector_dim", vector_dim))
    elif raw is not None:
        vector_cfg = getattr(raw, "vector_db", None)
        if vector_cfg is not None:
            db_path = str(getattr(vector_cfg, "db_path", db_path))
            vector_dim = int(getattr(vector_cfg, "vector_dim", vector_dim))

    return {
        "db_path": db_path,
        "vector_dim": vector_dim,
    }


def _build_placeholder_embedding(
    object_id: str, model_path: str, vector_dim: int
) -> np.ndarray:
    """生成可复现的伪向量兜底，仅在嵌入模型不可用时使用。"""
    import hashlib

    seed_text = f"{object_id}|{model_path}"
    seed_bytes = hashlib.sha256(seed_text.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(vector_dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    return vec


def _get_embedding_client():
    """获取 Qwen3-VL-Embedding 客户端单例，按需从配置初始化。"""
    from ai_modules.object_recognition.configs.dataclasses import (
        EmbeddingModelConfig,
        RecognitionConfig,
    )
    from ai_modules.object_recognition.tools.client_embedding import (
        get_embedding_client,
    )

    cfg = get_ai_config()
    raw = getattr(cfg, "object_recognition", None)

    if isinstance(raw, dict):
        embedding_raw = raw.get("embedding", {}) or {}
        embedding_cfg = (
            EmbeddingModelConfig(**embedding_raw)
            if embedding_raw
            else EmbeddingModelConfig()
        )
    elif isinstance(raw, RecognitionConfig):
        embedding_cfg = raw.embedding
    else:
        embedding_cfg = EmbeddingModelConfig()

    return get_embedding_client(embedding_cfg)


# ---------------------------------------------------------------------------
# 节点格式化器（Formatters）— 将业务数据转换为前端 parts
# ---------------------------------------------------------------------------


def _format_model_result_parts(
    data: Dict[str, Any],
    state: ModelRetrievalWorkflowState,
) -> List[Dict[str, Any]]:
    """将模型检索/生成结果格式化为前端 parts。"""
    model_results = state.get("model_results", [])
    # 直接复用 format_result_node 已计算好的统计量，无需重复 sum()
    mr_stats = data.get("global_assets", {}).get("model_retrieval", {})
    retrieval_count = mr_stats.get("retrieval_count", 0)
    generation_count = mr_stats.get("generation_count", 0)
    error_count = mr_stats.get("error_count", 0)

    md_parts: List[str] = ["## 模型检索与生成结果\n"]

    for r in model_results:
        name = r.get("item_name", "未知")
        source = r.get("source", "")
        error = r.get("error", "")

        if error and not source:
            md_parts.append(f"### {name}\n- **状态**: ❌ 失败 — {error}\n")
        elif source == "retrieval":
            object_id = r.get("object_id", "")
            distance = r.get("distance", 0)
            md_parts.append(
                f"### {name}\n"
                f"- **来源**: 检索命中\n"
                f"- **模型 ID**: {object_id}\n"
                f"- **相似度距离**: {distance:.4f}\n"
            )
        elif source == "generation":
            model_path = r.get("model_path", "")
            register_status = r.get("register_status", "")
            if error:
                md_parts.append(
                    f"### {name}\n" f"- **来源**: 3D 生成\n" f"- **状态**: ⚠️ {error}\n"
                )
            else:
                register_line = (
                    f"- **入库状态**: {register_status}\n" if register_status else ""
                )
                md_parts.append(
                    f"### {name}\n"
                    f"- **来源**: 3D 生成\n"
                    f"- **模型文件**: {model_path}\n"
                    f"{register_line}"
                )
        else:
            md_parts.append(f"### {name}\n- **状态**: 未知\n")

    md_parts.append(
        f"\n---\n**汇总**: 共 {len(model_results)} 个物体 — "
        f"检索命中 {retrieval_count}, "
        f"3D 生成 {generation_count}, "
        f"失败 {error_count}"
    )

    return [
        {
            "content_type": "text",
            "content_text": "\n".join(md_parts),
            "content_url": "",
            "parameter": {},
        }
    ]


# ---------------------------------------------------------------------------
# Node 1: dispatch_node — 组装任务清单
# ---------------------------------------------------------------------------


@stream_output_node("integrated", _NO_OUTPUT)
def dispatch_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """从第一步的输出中组装每个物体的检索/生成任务。

    优先从 global_assets["multi_scene"] 读取上游工作流写入的结果，
    兼容直接注入 approved_elements / generated_images 的调用方式。

    在工作流内置测试模式下，优先使用样例中的 global_assets。
    """
    metadata = state.get("metadata", {})
    global_assets = state.get("global_assets", {}) or {}

    # 工作流内置测试模式：优先使用测试样例的 global_assets
    if metadata.get("workflow_test"):
        test_case_key = metadata.get("workflow_test_case", "default")
        test_data = _get_test_case(test_case_key)
        test_assets = test_data.get("global_assets", {})
        if test_assets:
            logger.info(
                "[Workflow][dispatch][TEST] 工作流测试模式，使用预定义 global_assets: "
                f"test_case={test_case_key}"
            )
            global_assets = test_assets

    multi_scene = global_assets.get("multi_scene", {}) or {}

    approved = multi_scene.get("approved_elements") or state.get(
        "approved_elements", []
    )
    generated_images: Dict[str, str] = multi_scene.get("generated_images") or state.get(
        "generated_images", {}
    )

    if not approved:
        return {"error": "无可处理的设计元素（第一步输出为空）"}

    tasks: List[Dict[str, str]] = []
    for idx, elem in enumerate(approved, start=1):
        name = elem.get("item_name", "")
        image_url = generated_images.get(name, "")
        if not image_url:
            logger.warning(f"[Workflow][dispatch] {name} 无生成图片，跳过")
            continue
        object_id = _normalize_object_id(name, idx)
        tasks.append(
            {
                "item_name": name,
                "object_id": object_id,
                "image_url": image_url,
                "image_prompt": elem.get("image_prompt", ""),
            }
        )

    if not tasks:
        return {"error": "所有物体均无生成图片，无法进行模型检索"}

    logger.info(f"[Workflow][dispatch] 组装 {len(tasks)} 个检索/生成任务")
    return {
        "intermediate": {
            **state.get("intermediate", {}),
            "retrieval_tasks": tasks,
        },
    }


# ---------------------------------------------------------------------------
# Node 2: retrieve_or_generate_node — 检索或生成 3D 模型
# ---------------------------------------------------------------------------


def _retrieve_single_item(task: Dict[str, Any], search_tool: Any) -> Dict[str, Any]:
    """处理单个物体检索阶段，返回命中结果或待生成任务。"""
    name = task["item_name"]
    object_id = task.get("object_id", "")
    image_url = task["image_url"]
    image_prompt = task.get("image_prompt", "")

    result: Dict[str, Any] = {
        "item_name": name,
        "object_id": object_id,
        "task_index": task.get("task_index", 0),
        "input_image_url": image_url,
    }

    if not search_tool:
        result.update(
            {
                "source": "pending_generation",
                "search_status": "tool_unavailable",
            }
        )
        return result

    started_at = time.perf_counter()
    logger.info(f"[Workflow][retrieve] {name} 开始检索")

    try:
        raw = search_tool.invoke(
            {
                "query_images": [image_url],
                "query_text": image_prompt,
                "top_k": 1,
            }
        )
        search_info = _parse_search_result(raw)
        matches = search_info.get("matches", [])
        search_error = search_info.get("error", "")
        elapsed = time.perf_counter() - started_at

        if search_error:
            logger.warning(
                f"[Workflow][retrieve] {name} 检索失败，将降级生成: "
                f"{search_error} (elapsed={elapsed:.2f}s)"
            )
            result.update(
                {
                    "source": "pending_generation",
                    "search_status": "error",
                    "search_error": search_error,
                }
            )
            return result

        if matches and matches[0].get("distance", 999) < SEARCH_DISTANCE_THRESHOLD:
            best = matches[0]
            result.update(
                {
                    "source": "retrieval",
                    "object_id": best.get("object_id", ""),
                    "name": best.get("name", ""),
                    "distance": best.get("distance", 0),
                    "search_elapsed_seconds": round(elapsed, 3),
                }
            )
            logger.info(
                f"[Workflow][retrieve] {name} 检索命中: "
                f"object_id={best.get('object_id')}, "
                f"distance={best.get('distance', 0):.4f}, "
                f"elapsed={elapsed:.2f}s"
            )
            return result

        best_distance = matches[0].get("distance", "N/A") if matches else "N/A"
        logger.info(
            f"[Workflow][retrieve] {name} 检索未命中"
            f"（最佳 distance={best_distance}, elapsed={elapsed:.2f}s）"
        )
        result.update(
            {
                "source": "pending_generation",
                "search_status": "miss",
                "best_distance": best_distance,
                "search_elapsed_seconds": round(elapsed, 3),
            }
        )
        return result
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        logger.warning(
            f"[Workflow][retrieve] {name} 检索异常，将降级生成: "
            f"{e} (elapsed={elapsed:.2f}s)"
        )
        result.update(
            {
                "source": "pending_generation",
                "search_status": "error",
                "search_error": str(e),
                "search_elapsed_seconds": round(elapsed, 3),
            }
        )
        return result


def _generate_single_item(task: Dict[str, Any], generate_tool: Any) -> Dict[str, Any]:
    """处理单个物体生成阶段。"""
    name = task["item_name"]
    object_id = task.get("object_id", "")
    image_url = task.get("input_image_url") or task.get("image_url", "")
    result: Dict[str, Any] = {
        "item_name": name,
        "object_id": object_id,
        "task_index": task.get("task_index", 0),
        "input_image_url": image_url,
    }

    search_error = str(task.get("search_error", "") or "").strip()

    if not generate_tool:
        error_message = "检索未命中且 3D 生成工具不可用"
        if search_error:
            error_message = f"检索失败且 3D 生成工具不可用: {search_error}"
        result.update({"source": "generation", "error": error_message})
        return result

    started_at = time.perf_counter()
    logger.info(f"[Workflow][generate] {name} 开始 3D 生成")

    try:
        raw = generate_tool.invoke(
            {
                "mode": "image_to_3d",
                "images": [image_url],
                "object_id": object_id,
            }
        )
        model_info = _parse_3d_result(raw)
        elapsed = time.perf_counter() - started_at

        if model_info.get("error"):
            error_message = str(model_info.get("error", "生成结果解析为空"))
            logger.error(
                f"[Workflow][generate] {name} 3D 生成失败: "
                f"{error_message} (elapsed={elapsed:.2f}s)"
            )
            result.update({"source": "generation", "error": error_message})
            if search_error:
                result["search_error"] = search_error
            return result

        result.update(
            {
                "source": "generation",
                "model_path": model_info.get("model_path", ""),
                "parameter": model_info.get("parameter", {}),
                "generation_elapsed_seconds": round(elapsed, 3),
            }
        )
        if search_error:
            result["search_error"] = search_error

        logger.info(
            f"[Workflow][generate] {name} 3D 模型生成完成: "
            f"{model_info.get('model_path', '')} (elapsed={elapsed:.2f}s)"
        )
        return result
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        logger.error(
            f"[Workflow][generate] {name} 3D 生成失败: {e} " f"(elapsed={elapsed:.2f}s)"
        )
        result.update({"source": "generation", "error": str(e)})
        if search_error:
            result["search_error"] = search_error
        return result


@stream_output_node("integrated", _NO_OUTPUT)
def retrieve_or_generate_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """先完成全部检索，再对未命中的物体并发生成 3D 模型。"""

    tasks = state.get("intermediate", {}).get("retrieval_tasks", [])
    if not tasks:
        return {"error": "无检索/生成任务"}

    search_tool = _get_search_tool()
    generate_tool = None

    if not search_tool:
        logger.warning("[Workflow][retrieve_or_generate] 检索工具不可用，将全部走生成")

    retrieval_results: List[Dict[str, Any]] = []
    pending_generation: List[Dict[str, Any]] = []

    indexed_tasks = [
        {**task, "task_index": task.get("task_index", index)}
        for index, task in enumerate(tasks, start=1)
    ]

    max_workers = min(len(indexed_tasks), SEARCH_MAX_WORKERS) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_retrieve_single_item, task, search_tool): task
            for task in indexed_tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                retrieved = future.result()
            except Exception as e:
                logger.error(
                    f"[Workflow][retrieve_or_generate] "
                    f"{task.get('item_name', '?')} 检索任务异常: {e}"
                )
                retrieved = {
                    "item_name": task.get("item_name", "未知"),
                    "object_id": task.get("object_id", ""),
                    "task_index": task.get("task_index", 0),
                    "input_image_url": task.get("image_url", ""),
                    "source": "pending_generation",
                    "search_status": "error",
                    "search_error": str(e),
                }

            if retrieved.get("source") == "retrieval":
                retrieval_results.append(retrieved)
            else:
                pending_generation.append(retrieved)

    generated_results: List[Dict[str, Any]] = []
    if pending_generation:
        generate_tool = _get_3d_generate_tool()
        if not generate_tool:
            logger.warning(
                "[Workflow][retrieve_or_generate] 3D 生成工具不可用，未命中项将返回错误"
            )

        max_workers = min(len(pending_generation), GENERATION_MAX_WORKERS) or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_generate_single_item, task, generate_tool): task
                for task in pending_generation
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    generated_results.append(future.result())
                except Exception as e:
                    logger.error(
                        f"[Workflow][retrieve_or_generate] "
                        f"{task.get('item_name', '?')} 生成任务异常: {e}"
                    )
                    generated_results.append(
                        {
                            "item_name": task.get("item_name", "未知"),
                            "object_id": task.get("object_id", ""),
                            "task_index": task.get("task_index", 0),
                            "input_image_url": task.get("input_image_url", ""),
                            "source": "generation",
                            "error": str(e),
                        }
                    )

    results = sorted(
        retrieval_results + generated_results,
        key=lambda item: item.get("task_index", 0),
    )

    logger.info(
        "[Workflow][retrieve_or_generate] 完成: " "检索命中 %s, 生成 %s, 失败 %s",
        sum(1 for r in results if r.get("source") == "retrieval"),
        sum(
            1 for r in results if r.get("source") == "generation" and not r.get("error")
        ),
        sum(1 for r in results if r.get("error")),
    )

    return {"model_results": results}


# ---------------------------------------------------------------------------
# Node 3: register_node — 入库登记（占位）
# ---------------------------------------------------------------------------


@stream_output_node("integrated", _NO_OUTPUT)
def register_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """入库登记节点。

    对每个生成成功的物体，调用 Qwen3-VL-Embedding 生成存储侧嵌入向量并写入向量数据库。
    有图片时将图片与文本一起融合为向量；无图片时以物体名称作为纯文本输入触发嵌入。
    嵌入模型不可用时自动降级为可复现占位向量，保证入库流程不中断。
    """

    model_results = state.get("model_results", [])
    if not model_results:
        return {}

    from ai_modules.object_recognition.tools.vector_db import VectorDB
    from ai_modules.three_d_generate.tools.model_tools import wait_for_mesh_ready

    cfg = _get_recognition_db_config()
    vector_db = VectorDB(
        db_path=cfg["db_path"],
        vector_dim=cfg["vector_dim"],
    )

    inserted_count = 0
    updated_count = 0
    failed_count = 0
    skipped_count = 0
    enriched_results: List[Dict[str, Any]] = []

    try:
        for idx, row in enumerate(model_results, start=1):
            item = dict(row)

            if row.get("source") != "generation" or row.get("error"):
                item["register_status"] = "skipped"
                skipped_count += 1
                enriched_results.append(item)
                continue

            object_id = row.get("object_id") or _normalize_object_id(
                row.get("item_name", ""), idx
            )
            model_path = row.get("model_path", "")
            parameter = (
                row.get("parameter", {})
                if isinstance(row.get("parameter"), dict)
                else {}
            )

            # 等待后台 mesh 下载完成（Event 机制，不使用超时轮询）
            if parameter.get("has_mesh_pending", False):
                wait_object_id = parameter.get("object_id") or object_id
                logger.info(
                    "[Workflow][register] %s 等待后台 mesh 下载完成...",
                    wait_object_id,
                )
                wait_for_mesh_ready(wait_object_id)
                logger.info(
                    "[Workflow][register] %s mesh 下载已完成",
                    wait_object_id,
                )

            image_paths: List[str] = []
            preview_paths = parameter.get("preview_paths", [])
            if isinstance(preview_paths, list):
                image_paths.extend([str(p) for p in preview_paths if p])

            input_image_url = row.get("input_image_url", "")
            if input_image_url:
                image_paths.append(str(input_image_url))

            # 去重并保持顺序
            seen = set()
            dedup_paths = []
            for p in image_paths:
                if p not in seen:
                    seen.add(p)
                    dedup_paths.append(p)

            # 调用真实嵌入模型；无图片时以 item_name 作为纯文本输入触发嵌入
            item_name = row.get("item_name", "")
            text_desc = item_name or object_id
            try:
                embedding = _get_embedding_client().embed_for_storage(
                    image_paths=dedup_paths,
                    text=text_desc,
                )
            except Exception as emb_exc:  # noqa: BLE001
                logger.warning(
                    "[Workflow][register] %s 嵌入模型调用失败，降级为占位向量: %s",
                    object_id,
                    emb_exc,
                )
                embedding = _build_placeholder_embedding(
                    object_id=object_id,
                    model_path=model_path,
                    vector_dim=cfg["vector_dim"],
                )

            try:
                existing = vector_db.get_object(object_id)
                if existing is None:
                    rowid = vector_db.insert_object(
                        object_id=object_id,
                        embedding=embedding,
                        name=item_name,
                        category="generated_3d",
                        image_paths=dedup_paths,
                        description=text_desc,
                    )
                    item["register_status"] = "inserted"
                    item["register_rowid"] = rowid
                    inserted_count += 1
                else:
                    updated = vector_db.update_object(
                        object_id=object_id,
                        embedding=embedding,
                        name=item_name,
                        category="generated_3d",
                        image_paths=dedup_paths,
                        description=text_desc,
                    )
                    if updated:
                        item["register_status"] = "updated"
                        updated_count += 1
                    else:
                        item["register_status"] = "failed"
                        item["register_error"] = "更新失败"
                        failed_count += 1
            except Exception as e:  # noqa: BLE001
                item["register_status"] = "failed"
                item["register_error"] = str(e)
                failed_count += 1

            item["object_id"] = object_id
            enriched_results.append(item)
    finally:
        vector_db.close()

    logger.info(
        "[Workflow][register] 完成: inserted=%s, updated=%s, skipped=%s, failed=%s",
        inserted_count,
        updated_count,
        skipped_count,
        failed_count,
    )

    return {
        "model_results": enriched_results,
        "intermediate": {
            **state.get("intermediate", {}),
            "register_inserted": inserted_count,
            "register_updated": updated_count,
            "register_skipped": skipped_count,
            "register_failed": failed_count,
        },
    }


# ---------------------------------------------------------------------------
# Node 4: format_result_node — 结果格式化与 global_assets 写入
# ---------------------------------------------------------------------------


@stream_output_node("integrated", _format_model_result_parts)
def format_result_node(state: ModelRetrievalWorkflowState) -> Dict[str, Any]:
    """汇总模型检索/生成结果，写入 global_assets 并通过 formatter 输出对话内容。"""
    model_results = state.get("model_results", [])

    retrieval_count = sum(1 for r in model_results if r.get("source") == "retrieval")
    generation_count = sum(
        1
        for r in model_results
        if r.get("source") == "generation" and not r.get("error")
    )
    error_count = sum(1 for r in model_results if r.get("error"))

    logger.info(
        f"[Workflow][format_result] 完成: "
        f"检索 {retrieval_count}, 生成 {generation_count}, 失败 {error_count}"
    )

    return {
        "global_assets": {
            "model_retrieval": {
                "model_results": model_results,
                "retrieval_count": retrieval_count,
                "generation_count": generation_count,
                "error_count": error_count,
            }
        },
        "intermediate": {
            **state.get("intermediate", {}),
            "workflow": "model_retrieval",
            "retrieval_count": retrieval_count,
            "generation_count": generation_count,
            "error_count": error_count,
        },
    }


# ---------------------------------------------------------------------------
# DAG 构建与导出
# ---------------------------------------------------------------------------


def build_model_retrieval_workflow() -> "CompiledStateGraph":
    """构建模型检索与生成 LangGraph DAG。

    拓扑：
        START → dispatch → retrieve_or_generate → register → format_result → END
    """
    graph = StateGraph(ModelRetrievalWorkflowState)

    graph.add_node("dispatch", dispatch_node)
    graph.add_node("retrieve_or_generate", retrieve_or_generate_node)
    graph.add_node("register", register_node)
    graph.add_node("format_result", format_result_node)

    graph.add_edge(START, "dispatch")
    graph.add_edge("dispatch", "retrieve_or_generate")
    graph.add_edge("retrieve_or_generate", "register")
    graph.add_edge("register", "format_result")
    graph.add_edge("format_result", END)

    return graph.compile()


WORKFLOWS: Dict[int, "CompiledStateGraph"] = {
    MODEL_RETRIEVAL_FUNCTION_ID: build_model_retrieval_workflow(),
}

WORKFLOW_COMMANDS: Dict[str, int] = {
    "/model_retrieval": MODEL_RETRIEVAL_FUNCTION_ID,
}

__all__ = [
    "WORKFLOWS",
    "WORKFLOW_COMMANDS",
    "MODEL_RETRIEVAL_FUNCTION_ID",
    "build_model_retrieval_workflow",
]
