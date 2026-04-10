"""compose_scene 节点 — 调用 place_scene_from_items 工具生成 scene.json。"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import get_tool, parse_placement_result

logger = logging.getLogger(__name__)


@stream_output_node("integrated", NO_OUTPUT)
def compose_scene_node(state) -> Dict[str, Any]:
    """调用 place_scene_from_items 生成场景布局文件。"""
    intermediate = state.get("intermediate", {})
    placement_items = intermediate.get("placement_items", [])
    metadata = state.get("metadata", {})

    if not placement_items:
        return {"error": "placement_items 为空，无法组合场景"}

    # 场景参数，支持从 metadata 或 prompt 中获取
    scene_name = metadata.get("scene_name", "composed_scene")
    scene_path = metadata.get("scene_path", f"Scene/{scene_name}/{scene_name}.scene")
    room_size = metadata.get("room_size", [5, 3, 5])

    tool = get_tool("place_scene_from_items")
    if tool is None:
        return {"error": "place_scene_from_items 工具未注册"}

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
