"""场景级多视图拍摄 Agent 工具

围绕整个场景 AABB 球面生成多组相机位姿，逐个移动相机并截图。
同时输出 transforms.json（NeRF / 3DGS 兼容格式）。
"""
from __future__ import annotations

import datetime
import json
import math
import os
import time
from typing import List, Optional, Tuple

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)

DEFAULT_SCENE_NAME = ""


def _resolve_scene(scene_manager, scene_name: str):
    if scene_name:
        scene = scene_manager.get(scene_name)
        if scene is not None:
            return scene
        for route in scene_manager.list_all():
            s = scene_manager.get(route)
            if s is not None and getattr(s, "name", None) == scene_name:
                return s
    routes = scene_manager.list_all()
    if routes:
        return scene_manager.get(routes[0])
    return None


# ===========================================================================
# Input Schema
# ===========================================================================

class SceneMultiViewInput(BaseModel):
    scene_name: str = Field(
        default=DEFAULT_SCENE_NAME,
        description="目标场景名称，为空则使用当前场景",
    )
    camera_name: str | None = Field(
        default=None,
        description="摄像头名称，为空则使用主摄像头",
    )
    num_views_per_ring: int = Field(
        default=12,
        description="每个仰角环上均匀采样的视角数量，默认 12",
    )
    elevations_deg: List[float] = Field(
        default=[30.0, 0.0, -15.0],
        description="仰角列表（度）。正值俯瞰，负值仰拍。默认 [30, 0, -15]",
    )
    radius_scale: float = Field(
        default=2.0,
        description="相机距场景中心的距离 = 场景半径 × radius_scale，默认 2.0",
    )
    fov: float = Field(
        default=60.0,
        description="拍摄视场角（度），默认 60",
    )
    output_dir: str | None = Field(
        default=None,
        description="输出目录路径。为空则自动生成到项目 screenshots/ 目录下",
    )
    output_modes: List[str] = Field(
        default=["final_color"],
        description="每个视角要截取的输出通道列表，可选: final_color, base_color, normal, position, object_id",
    )


# ===========================================================================
# Helpers
# ===========================================================================

def _get_screenshot_dir() -> str:
    from config.paths_config import get_project_screenshots_dir
    return str(get_project_screenshots_dir())


def _build_c2w_matrix(
    position: List[float],
    forward: List[float],
    world_up: List[float],
) -> List[List[float]]:
    """构建 4×4 camera-to-world 矩阵。"""
    zx, zy, zz = forward
    ux, uy, uz = world_up

    # right = normalize(cross(up, forward))
    rx = uy * zz - uz * zy
    ry = uz * zx - ux * zz
    rz = ux * zy - uy * zx
    rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rlen > 1e-7:
        rx /= rlen; ry /= rlen; rz /= rlen

    # true_up = cross(forward, right)
    tux = zy * rz - zz * ry
    tuy = zz * rx - zx * rz
    tuz = zx * ry - zy * rx

    px, py, pz = position
    return [
        [rx, tux, zx, px],
        [ry, tuy, zy, py],
        [rz, tuz, zz, pz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _write_transforms_json(
    output_dir: str,
    poses: List[dict],
    fov: float,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """写出 NeRF / 3DGS 兼容的 transforms.json，返回文件路径。"""
    camera_angle_x = math.radians(fov)
    frames = []
    for idx, p in enumerate(poses):
        frames.append({
            "file_path": f"view_{idx:04d}.png",
            "transform_matrix": _build_c2w_matrix(
                p["position"], p["forward"], p["world_up"],
            ),
        })

    data = {
        "camera_angle_x": camera_angle_x,
        "w": width,
        "h": height,
        "frames": frames,
    }
    out_path = os.path.join(output_dir, "transforms.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


def _generate_sphere_poses(
    center: Tuple[float, float, float],
    radius: float,
    num_views_per_ring: int,
    elevations_deg: List[float],
    fov: float,
) -> List[dict]:
    """在球面上生成围绕 center 的相机位姿列表。"""
    cx, cy, cz = center
    poses: List[dict] = []

    for elev_deg in elevations_deg:
        elev_rad = math.radians(elev_deg)
        y_offset = radius * math.sin(elev_rad)
        horiz_r = radius * math.cos(elev_rad)

        for i in range(num_views_per_ring):
            angle = 2.0 * math.pi * i / num_views_per_ring
            px = cx + horiz_r * math.cos(angle)
            py = cy + y_offset
            pz = cz + horiz_r * math.sin(angle)

            fx, fy, fz = cx - px, cy - py, cz - pz
            length = math.sqrt(fx * fx + fy * fy + fz * fz)
            if length > 1e-7:
                fx /= length; fy /= length; fz /= length

            poses.append({
                "position": [px, py, pz],
                "forward": [fx, fy, fz],
                "world_up": [0.0, 1.0, 0.0],
                "fov": fov,
                "elevation_deg": elev_deg,
                "azimuth_deg": math.degrees(angle),
            })

    return poses


# ===========================================================================
# Tool builder
# ===========================================================================

def _build_scene_multi_view_tool(scene_manager) -> StructuredTool:
    """构建场景级多视图拍摄工具"""

    def _scene_multi_view(
        *,
        scene_name: str = DEFAULT_SCENE_NAME,
        camera_name: str | None = None,
        num_views_per_ring: int = 12,
        elevations_deg: List[float] | None = None,
        radius_scale: float = 2.0,
        fov: float = 60.0,
        output_dir: str | None = None,
        output_modes: List[str] | None = None,
    ) -> str:
        try:
            if elevations_deg is None:
                elevations_deg = [30.0, 0.0, -15.0]
            if output_modes is None:
                output_modes = ["final_color"]

            # --- 解析场景和相机 ---
            scene = _resolve_scene(scene_manager, scene_name)
            if scene is None:
                return build_error_result(
                    error_message="No scene loaded",
                ).to_envelope(interface_type="scene")

            camera = scene.find_camera(camera_name)
            if camera is None:
                return build_error_result(
                    error_message=f"No camera available in scene '{scene_name}'",
                ).to_envelope(interface_type="scene")

            # --- 计算场景 AABB 和中心 ---
            aabb = scene.get_aabb()  # [min_x, min_y, min_z, max_x, max_y, max_z]
            cx = (aabb[0] + aabb[3]) * 0.5
            cy = (aabb[1] + aabb[4]) * 0.5
            cz = (aabb[2] + aabb[5]) * 0.5

            dx = aabb[3] - aabb[0]
            dy = aabb[4] - aabb[1]
            dz = aabb[5] - aabb[2]
            scene_radius = math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5
            if scene_radius < 1e-6:
                scene_radius = 5.0

            camera_radius = scene_radius * radius_scale

            # --- 生成位姿 ---
            poses = _generate_sphere_poses(
                center=(cx, cy, cz),
                radius=camera_radius,
                num_views_per_ring=num_views_per_ring,
                elevations_deg=elevations_deg,
                fov=fov,
            )
            if not poses:
                return build_error_result(
                    error_message="No camera poses generated",
                ).to_envelope(interface_type="scene")

            # --- 输出目录 ---
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not output_dir:
                output_dir = os.path.join(
                    _get_screenshot_dir(), f"scene_multiview_{ts}",
                )
            elif not os.path.isabs(output_dir):
                output_dir = os.path.join(_get_screenshot_dir(), output_dir)
            os.makedirs(output_dir, exist_ok=True)

            # --- 保存原始相机状态 ---
            orig_pos = list(camera.get_position())
            orig_fwd = list(camera.get_forward())
            orig_up = list(camera.get_world_up())
            orig_fov = camera.get_fov()
            prev_mode = camera.get_output_mode()

            # --- 写 transforms.json ---
            transforms_path = _write_transforms_json(output_dir, poses, fov)

            # --- 逐个位姿拍摄 ---
            saved_files: List[str] = []
            total = len(poses)

            for idx, pose in enumerate(poses):
                camera.set(
                    pose["position"],
                    pose["forward"],
                    pose["world_up"],
                    pose["fov"],
                )
                time.sleep(0.15)

                az_deg = int(pose["azimuth_deg"])
                el_deg = int(pose["elevation_deg"])

                for mode in output_modes:
                    if mode != camera.get_output_mode():
                        camera.set_output_mode(mode)
                        time.sleep(0.1)

                    filename = f"view_{idx:04d}_el{el_deg:+03d}_az{az_deg:03d}_{mode}.png"
                    filepath = os.path.join(output_dir, filename)
                    camera.save_screenshot(filepath)
                    saved_files.append(filepath)

            # --- 恢复原始相机 ---
            camera.set(orig_pos, orig_fwd, orig_up, orig_fov)
            if camera.get_output_mode() != prev_mode:
                camera.set_output_mode(prev_mode)

            result_data = {
                "status": "success",
                "scene_center": [cx, cy, cz],
                "scene_radius": scene_radius,
                "camera_radius": camera_radius,
                "num_views_per_ring": num_views_per_ring,
                "elevations_deg": elevations_deg,
                "total_poses": total,
                "output_modes": output_modes,
                "total_images": len(saved_files),
                "output_dir": output_dir,
                "transforms_json": transforms_path,
                "files": saved_files,
            }
            part = build_part(
                content_type="text",
                content_text=json.dumps(result_data, ensure_ascii=False),
            )
            return build_success_result(parts=[part]).to_envelope(
                interface_type="scene"
            )
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="scene"
            )

    return StructuredTool(
        name="scene_multi_view_capture",
        description=(
            "对整个场景进行多视图环绕拍摄。"
            "基于场景 AABB 自动计算中心和半径，在球面上按多个仰角环均匀生成相机位姿，"
            "逐个移动相机拍摄截图，并输出 transforms.json（兼容 NeRF / 3DGS）。"
            "适用于 3D 重建数据采集、全景展示等场景。"
        ),
        args_schema=SceneMultiViewInput,
        func=_scene_multi_view,
    )


# ===========================================================================
# Loader
# ===========================================================================

def load_multi_view_tools() -> List[StructuredTool]:
    from CoronaCore.core.managers import scene_manager
    return [
        _build_scene_multi_view_tool(scene_manager),
    ]


__all__ = ["load_multi_view_tools"]
