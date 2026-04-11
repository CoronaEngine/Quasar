"""compose_scene 节点 — 调用 LLM 进行智能布局，再生成 scene.json。"""

from __future__ import annotations

import json
import logging
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

坐标系说明：
- X 轴：左右方向（正方向向右）
- Y 轴：上下方向（正方向向上），大多数物体 Y=0 表示放在地面上
- Z 轴：前后方向（正方向向前）
- 旋转单位为角度（degree），绕各轴旋转
- 房间中心为原点 (0, 0, 0)

布局原则：
1. 根据物体的语义功能决定摆放位置（如：床放卧室中央、桌子靠墙、椅子在桌旁）
2. 物体之间保持合理间距，避免重叠
3. 物体朝向应符合使用习惯（如：椅子面向桌子、沙发面向电视）
4. 考虑用户描述中的空间关系（如"靠墙"、"居中"、"对称"等）
5. 所有物体必须在房间范围内（pos 的 X/Z 绝对值不超过房间对应半径）
6. 缩放默认为 [1,1,1]，除非用户明确要求大小变化

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
    return (
        f"## 设计方案\n{prompt}\n\n"
        f"## 房间尺寸\n长(X)={room_size[0]}m, 高(Y)={room_size[1]}m, 宽(Z)={room_size[2]}m\n\n"
        f"## 物体列表（共 {len(items)} 个）\n" + "\n".join(item_lines)
    )


def _call_llm_for_layout(
    prompt: str,
    room_size: List[float],
    items: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """调用 LLM 生成智能布局，返回布局列表；失败时返回 None。"""
    try:
        from ai_models.base_pool.registry import get_chat_model
        llm = get_chat_model(temperature=0.3, request_timeout=60.0)
    except Exception as e:
        logger.warning("compose_scene: 无法获取 LLM，使用默认布局: %s", e)
        return None

    user_prompt = _build_layout_user_prompt(prompt, room_size, items)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke([
            SystemMessage(content=_LAYOUT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        text = (response.content if hasattr(response, "content") else str(response)).strip()

        # 兼容 markdown 代码块包裹
        if "```" in text:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                text = text[start: end + 1]

        layouts: List[Dict[str, Any]] = json.loads(text)
        if not isinstance(layouts, list):
            logger.warning("compose_scene: LLM 返回非数组: %s", type(layouts))
            return None

        for entry in layouts:
            if not isinstance(entry, dict) or "object_id" not in entry:
                logger.warning("compose_scene: LLM 布局条目格式异常: %s", entry)
                return None

        logger.info("compose_scene: LLM 智能布局成功，生成 %d 个物体位置", len(layouts))
        return layouts

    except json.JSONDecodeError as e:
        logger.warning("compose_scene: LLM 返回 JSON 解析失败: %s", e)
        return None
    except Exception as e:
        logger.warning("compose_scene: LLM 布局调用失败: %s", e)
        return None


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

    if not placement_items:
        return {"error": "placement_items 为空，无法组合场景"}

    # 场景参数
    scene_name = metadata.get("scene_name", "composed_scene")
    scene_path = metadata.get("scene_path", f"Scene/{scene_name}/{scene_name}.scene")
    room_size = metadata.get("room_size", [5, 3, 5])
    prompt = state.get("prompt", "")

    tool = get_tool("place_scene_from_items")
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
