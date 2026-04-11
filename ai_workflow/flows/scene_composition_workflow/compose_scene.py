"""compose_scene 节点 — 调用 LLM 进行智能布局，再生成 scene.json。"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import get_tool, parse_placement_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM 智能布局
# ---------------------------------------------------------------------------

_LAYOUT_SYSTEM_PROMPT = """\
你是一个专业的 3D 场景布局规划师。根据用户的设计方案、房间尺寸和物体列表，
为每个物体生成合理的摆放位置（pos）、旋转（rot）和缩放（scale）。

坐标系说明（引擎标准坐标系）：
- X 轴：左右方向（正方向向右）
- Y 轴：高度方向（正方向向上），Y=0 为地面
- Z 轴：深度方向（正方向向屏幕内/向北）
- 旋转单位为角度（degree），绕 Y 轴旋转可控制朝向
- 房间中心为原点 (0, 0, 0)，物体放置在 XZ 平面上

布局原则：
1. 根据物体的语义功能决定摆放位置（如：床放卧室中央、桌子靠墙、椅子在桌旁）
2. 物体之间保持合理间距，避免重叠
3. 物体朝向应符合使用习惯（如：椅子面向桌子、沙发面向电视）
4. 考虑用户描述中的空间关系（如"靠墙"、"居中"、"对称"等）
5. 所有物体必须在房间范围内，用户消息中会提供具体的 X/Z 边界，严格遵守
6. 缩放默认为 [1,1,1]，除非用户明确要求大小变化
7. 大多数物体 Y=0（放在地面），悬挂物（灯、画）可设置 Y>0

你必须且只能返回一个 JSON 数组，格式如下（不要包含其他文字）：
[
  {
    "object_id": "物体ID",
    "pos": [x, y, z],
    "rot": [rx, ry, rz],
    "scale": [sx, sy, sz]
  }
]
"""


def _build_layout_user_prompt(
    prompt: str,
    room_size: List[float],
    items: List[Dict[str, Any]],
) -> str:
    """构建发送给 LLM 的用户 prompt。"""
    item_lines = [
        f"  - object_id: {it.get('object_id', '')}, 名称: {it.get('name', '未知')}"
        for it in items
    ]
    x_half = room_size[0] / 2.0
    z_half = room_size[1] / 2.0
    y_height = room_size[2] if len(room_size) > 2 else 3.0
    return (
        f"## 设计方案\n{prompt}\n\n"
        f"## 房间尺寸\n"
        f"长(X轴)={room_size[0]}m 范围[{-x_half:.1f}, {x_half:.1f}], "
        f"宽(Z轴)={room_size[1]}m 范围[{-z_half:.1f}, {z_half:.1f}], "
        f"高(Y轴)={y_height}m\n地面坐标 Y=0, 天花板 Y={y_height}\n\n"
        f"## 物体列表（共 {len(items)} 个）\n" + "\n".join(item_lines)
    )


def _call_llm_for_layout(
    prompt: str,
    room_size: List[float],
    items: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """调用 LLM 生成智能布局，返回布局列表；失败时返回 None。"""
    LLM_TOTAL_TIMEOUT = 50.0  # 主线程等待超时（秒）

    user_prompt = _build_layout_user_prompt(prompt, room_size, items)
    logger.info(
        "compose_scene: 启动 LLM 布局线程（超时 %.0fs，%d 个物体）...",
        LLM_TOTAL_TIMEOUT,
        len(items),
    )

    def _do_llm_call():
        """在后台线程中完成 get_chat_model + invoke，避免任何阻塞传到主线程。"""
        from ai_models.base_pool.registry import get_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        logger.info("compose_scene: [worker] 正在获取 LLM 客户端...")
        llm = get_chat_model(temperature=0.3, request_timeout=45.0)
        logger.info("compose_scene: [worker] LLM 就绪，调用 invoke...")
        return llm.invoke([
            SystemMessage(content=_LAYOUT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

    # 不使用 with 语句，避免 __exit__ 的 shutdown(wait=True) 在超时后继续阻塞
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_do_llm_call)
    try:
        response = future.result(timeout=LLM_TOTAL_TIMEOUT)
    except FuturesTimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        logger.warning("compose_scene: LLM 调用超时（%.0fs），使用默认布局", LLM_TOTAL_TIMEOUT)
        return None
    except Exception as e:
        executor.shutdown(wait=False)
        logger.warning("compose_scene: LLM 调用失败: %s", e)
        return None
    else:
        executor.shutdown(wait=False)

    text = (response.content if hasattr(response, "content") else str(response)).strip()

    # 兼容 markdown 代码块包裹
    if "```" in text:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start: end + 1]

    try:
        layouts: List[Dict[str, Any]] = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("compose_scene: LLM 返回 JSON 解析失败: %s", e)
        return None

    if not isinstance(layouts, list):
        logger.warning("compose_scene: LLM 返回非数组: %s", type(layouts))
        return None

    for entry in layouts:
        if not isinstance(entry, dict) or "object_id" not in entry:
            logger.warning("compose_scene: LLM 布局条目格式异常: %s", entry)
            return None

    logger.info("compose_scene: LLM 智能布局成功，生成 %d 个物体位置", len(layouts))
    return layouts


def _apply_llm_layout(
    placement_items: List[Dict[str, Any]],
    layouts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """将 LLM 生成的布局覆盖到 placement_items 对应条目上。"""
    layout_map = {str(l["object_id"]): l for l in layouts}
    for item in placement_items:
        oid = str(item.get("object_id", ""))
        layout = layout_map.get(oid)
        if layout is None:
            continue
        for key in ("pos", "rot", "scale"):
            val = layout.get(key)
            if isinstance(val, list) and len(val) == 3:
                item[key] = [float(v) for v in val]
    return placement_items


# ---------------------------------------------------------------------------
# 节点入口
# ---------------------------------------------------------------------------


@stream_output_node("integrated", NO_OUTPUT)
def compose_scene_node(state) -> Dict[str, Any]:
    """调用 LLM 智能布局 + place_scene_from_items 生成场景布局文件。"""
    intermediate = state.get("intermediate", {})
    placement_items = intermediate.get("placement_items", [])
    metadata = state.get("metadata", {})
    prompt = state.get("prompt", "")
    logger.info(
        "compose_scene: 节点启动 (items=%d, prompt=%d 字符)",
        len(placement_items),
        len(prompt),
    )

    if not placement_items:
        return {"error": "placement_items 为空，无法组合场景"}

    # 场景参数
    scene_name = metadata.get("scene_name", "composed_scene")
    scene_path = metadata.get("scene_path", f"Scene/{scene_name}/{scene_name}.scene")
    room_size = metadata.get("room_size", [5, 3, 5])

    tool = get_tool("place_scene_from_items")
    logger.info("compose_scene: 工具获取完成 (found=%s)", tool is not None)
    if tool is None:
        return {"error": "place_scene_from_items 工具未注册"}

    # ---- 智能布局：用 LLM 生成位置覆盖 ----
    if prompt:
        layouts = _call_llm_for_layout(prompt, room_size, placement_items)
        if layouts:
            placement_items = _apply_llm_layout(list(placement_items), layouts)
            logger.info("compose_scene: 已应用 LLM 智能布局")
        else:
            logger.info("compose_scene: LLM 布局失败，回退到默认确定性布局")
    else:
        logger.info("compose_scene: 无用户设计方案，使用默认确定性布局")

    logger.info(
        "compose_scene: 调用 place_scene_from_items (items=%d, room=%s)",
        len(placement_items),
        room_size,
    )

    raw_result = tool.invoke({
        "scene_path": scene_path,
        "scene_name": scene_name,
        "room_size": room_size,
        "items": placement_items,
    })

    parsed = parse_placement_result(raw_result)
    if parsed.get("error"):
        return {"error": f"场景布局失败: {parsed['error']}"}

    scene_json_path = parsed["scene_path"]
    actors = parsed.get("actors", [])

    logger.info("compose_scene: scene.json 已生成 → %s (%d actors)", scene_json_path, len(actors))

    return {
        "intermediate": {
            "scene_json_path": scene_json_path,
            "scene_actors": actors,
            "scene_name": scene_name,
        },
    }
