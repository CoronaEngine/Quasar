"""capture_screenshots 节点 — 模型导入后对场景进行多角度拍摄，为 review 准备截图。"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_workflow.streaming import stream_output_node

from .formatters import NO_OUTPUT
from .helpers import get_tool

logger = logging.getLogger(__name__)

# 场景全貌环绕拍摄参数（以房间中心为目标）
_SCENE_VIEW_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]  # 水平方位角（度）
_ELEVATION_DEG = 35.0   # 俯视仰角
_SCENE_DISTANCE_FACTOR = 1.15  # 相对房间半径的观察距离倍数


def _calc_camera_pose(
    center: List[float],
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> Dict[str, List[float]]:
    """根据球面坐标计算摄像机 position 和 forward。"""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    pos = [
        center[0] + distance * cos_el * math.sin(az),
        center[1] + distance * math.sin(el),
        center[2] + distance * cos_el * math.cos(az),
    ]
    fwd_raw = [center[0] - pos[0], center[1] - pos[1], center[2] - pos[2]]
    length = math.sqrt(sum(f * f for f in fwd_raw))
    fwd = [f / length for f in fwd_raw] if length > 1e-6 else [0.0, 0.0, -1.0]
    return {"position": pos, "forward": fwd, "up": [0.0, 1.0, 0.0]}


def _parse_tool_json(raw: Any) -> Dict[str, Any]:
    """从工具 envelope 解析 JSON content_text。"""
    try:
        envelope = json.loads(raw) if isinstance(raw, str) else raw
        parts = (
            envelope.get("result", {}).get("parts")
            or (envelope.get("llm_content") or [{}])[0].get("part", [])
        )
        for part in (parts or []):
            text = part.get("content_text", "")
            if text:
                return json.loads(text)
    except Exception:
        pass
    return {}


@stream_output_node("integrated", NO_OUTPUT)
def capture_screenshots_node(state) -> Dict[str, Any]:
    """导入完成后，从多角度拍摄场景全貌截图，存入 review_screenshots/ 供后续审查使用。"""
    intermediate = state.get("intermediate", {})
    scene_json_path = intermediate.get("scene_json_path", "")
    imported_actors = intermediate.get("imported_actors", [])
    metadata = state.get("metadata", {})
    room_size = metadata.get("room_size", [5, 5, 3])

    if not imported_actors:
        logger.info("capture_screenshots: 无导入成功的 actor，跳过截图")
        return {}

    if not scene_json_path:
        logger.warning("capture_screenshots: scene_json_path 为空，跳过截图")
        return {}

    # 截图输出目录
    output_dir = str(Path(scene_json_path).parent / "review_screenshots")
    os.makedirs(output_dir, exist_ok=True)

    # 观察距离 = 房间最长边的一半 * 系数
    x_half = room_size[0] / 2.0
    z_half = room_size[1] / 2.0 if len(room_size) > 1 else x_half
    distance = max(x_half, z_half) * 2.0 * _SCENE_DISTANCE_FACTOR
    center = [0.0, room_size[2] / 4.0 if len(room_size) > 2 else 0.5, 0.0]

    # 需要摄像机控制工具
    move_tool = get_tool("camera_move")
    shot_tool = get_tool("camera_screenshot")

    if move_tool is None or shot_tool is None:
        logger.warning(
            "capture_screenshots: camera_move=%s camera_screenshot=%s，工具缺失，跳过截图",
            move_tool is not None,
            shot_tool is not None,
        )
        return {}

    logger.info(
        "capture_screenshots: 开始拍摄 %d 个视角 → %s",
        len(_SCENE_VIEW_ANGLES),
        output_dir,
    )

    saved: List[str] = []
    for az in _SCENE_VIEW_ANGLES:
        pose = _calc_camera_pose(center, distance, az, _ELEVATION_DEG)
        try:
            move_tool.invoke({
                "position": pose["position"],
                "forward": pose["forward"],
                "up": pose["up"],
            })
            time.sleep(0.2)  # 等渲染稳定

            filename = f"scene_az{az:03d}.png"
            filepath = os.path.join(output_dir, filename)
            raw = shot_tool.invoke({"output_path": filepath})
            result = _parse_tool_json(raw)
            if result.get("status") == "success":
                saved.append(filepath)
                logger.debug("capture_screenshots: 已保存 %s", filepath)
            else:
                logger.warning("capture_screenshots: az=%d 截图失败: %s", az, result)
        except Exception as exc:
            logger.warning("capture_screenshots: az=%d 异常: %s", az, exc)

    logger.info(
        "capture_screenshots: 完成，共保存 %d/%d 张截图",
        len(saved),
        len(_SCENE_VIEW_ANGLES),
    )

    return {
        "intermediate": {
            "review_screenshot_dir": output_dir,
            "review_screenshot_count": len(saved),
        },
    }
