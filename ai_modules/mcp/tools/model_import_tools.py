"""
模型导入 / 删除工具

提供将本地模型文件导入到当前场景以及从场景中删除模型的能力。
支持 .obj / .dae / .glb / .gltf / .fbx 格式。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)

DEFAULT_SCENE_NAME = ""
SUPPORTED_EXTS = {".obj", ".dae", ".glb", ".gltf", ".fbx"}


def _resolve_scene(scene_manager, scene_name: str):
    """根据名称获取场景，若为空则自动获取当前已加载的场景。"""
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


def _pick_model_file(path: str) -> Optional[str]:
    """
    如果 path 是目录，尝试从中挑选第一个支持的模型文件；
    如果 path 是文件，直接返回。
    """
    if os.path.isfile(path):
        if Path(path).suffix.lower() in SUPPORTED_EXTS:
            return path
        return None

    if os.path.isdir(path):
        for ext in SUPPORTED_EXTS:
            for f in sorted(os.listdir(path)):
                if f.lower().endswith(ext):
                    return os.path.join(path, f)
    return None


# ---------------------------------------------------------------------------
# Input Schema
# ---------------------------------------------------------------------------

class ImportModelInput(BaseModel):
    """将本地模型文件导入到当前场景"""

    model_path: str = Field(
        description="模型文件的路径（绝对路径或项目相对路径），支持 .obj/.dae/.glb/.gltf/.fbx。"
                    "也可以传入包含模型文件的目录路径，会自动选取其中的模型文件。",
    )
    actor_name: Optional[str] = Field(
        default=None,
        description="导入后在场景中的名称，为空则使用文件名",
    )
    position: Optional[List[float]] = Field(
        default=None,
        description="初始位置 [x, y, z]，为空默认 [0, 0, 0]",
    )
    rotation: Optional[List[float]] = Field(
        default=None,
        description="初始旋转（欧拉角）[pitch, yaw, roll]，为空默认 [0, 0, 0]",
    )
    scale: Optional[List[float]] = Field(
        default=None,
        description="初始缩放 [sx, sy, sz]，为空默认 [1, 1, 1]",
    )
    scene_name: str = Field(
        default=DEFAULT_SCENE_NAME,
        description="目标场景名称，为空则使用当前场景",
    )


class RemoveModelInput(BaseModel):
    """从场景中删除指定模型"""

    actor_name: str = Field(
        description="要删除的模型（Actor）名称",
    )
    scene_name: str = Field(
        default=DEFAULT_SCENE_NAME,
        description="目标场景名称，为空则使用当前场景",
    )


# ---------------------------------------------------------------------------
# Tool Builder
# ---------------------------------------------------------------------------

def _build_import_model_tool(scene_manager) -> StructuredTool:
    """构建模型导入工具"""

    def _import_model(
        *,
        model_path: str,
        actor_name: str | None = None,
        position: List[float] | None = None,
        rotation: List[float] | None = None,
        scale: List[float] | None = None,
        scene_name: str = DEFAULT_SCENE_NAME,
    ) -> str:
        try:
            from CoronaCore.core.corona_editor import CoronaEditor
            CoronaEngine = CoronaEditor.CoronaEngine

            # 1. 解析场景
            scene = _resolve_scene(scene_manager, scene_name)
            if scene is None:
                return build_error_result(
                    error_message="没有已加载的场景，请先打开或创建一个场景"
                ).to_envelope(interface_type="scene")

            # 2. 解析模型路径（支持绝对路径和项目相对路径）
            if os.path.isabs(model_path):
                resolved_path = model_path
            else:
                project_path = CoronaEngine.active_project_path
                if not project_path:
                    return build_error_result(
                        error_message="未设置活跃项目路径，无法解析相对路径"
                    ).to_envelope(interface_type="scene")
                resolved_path = os.path.join(project_path, model_path)

            # 3. 如果是目录，尝试挑选模型文件
            final_path = _pick_model_file(resolved_path)
            if final_path is None:
                return build_error_result(
                    error_message=f"找不到支持的模型文件: {resolved_path}，"
                                  f"支持格式: {', '.join(sorted(SUPPORTED_EXTS))}"
                ).to_envelope(interface_type="scene")

            if not os.path.exists(final_path):
                return build_error_result(
                    error_message=f"模型文件不存在: {final_path}"
                ).to_envelope(interface_type="scene")

            # 4. 确定 actor 名称（自动处理同名冲突）
            base_name = actor_name or Path(final_path).stem
            existing_names = {a.name for a in scene.get_actors()}
            name = base_name
            suffix = 1
            while name in existing_names:
                name = f"{base_name}_{suffix}"
                suffix += 1

            # 5. 创建 Actor 并加载模型
            from CoronaCore.core.entities.actor import Actor

            actor = Actor(
                name=name,
                route=final_path,
                actor_type="mesh",
                parent_scene=scene,
            )

            # 6. 设置变换
            if position:
                actor.set_position(position)
            if rotation:
                actor.set_rotation(rotation)
            if scale:
                actor.set_scale(scale)

            # 7. 添加到场景
            scene.add_actor(actor)

            # 8. 构建返回结果
            result_data = {
                "status": "success",
                "actor_name": actor.name,
                "model_path": final_path,
                "position": actor.get_position(),
                "rotation": actor.get_rotation(),
                "scale": actor.get_scale(),
                "scene": getattr(scene, "name", scene_name),
            }
            part = build_part(
                content_type="text",
                content_text=json.dumps(result_data, ensure_ascii=False),
            )
            return build_success_result(parts=[part]).to_envelope(
                interface_type="scene"
            )

        except FileNotFoundError as e:
            return build_error_result(
                error_message=str(e)
            ).to_envelope(interface_type="scene")
        except Exception as e:
            return build_error_result(
                error_message=f"模型导入失败: {e}"
            ).to_envelope(interface_type="scene")

    return StructuredTool(
        name="import_model",
        description="将本地 3D 模型文件导入到当前场景中。"
                    "支持 .obj/.dae/.glb/.gltf/.fbx 格式。"
                    "可指定名称、位置、旋转、缩放等参数。"
                    "也可传入包含模型文件的目录路径，会自动选取其中的模型文件。",
        args_schema=ImportModelInput,
        func=_import_model,
    )


def _build_remove_model_tool(scene_manager) -> StructuredTool:
    """构建模型删除工具"""

    def _remove_model(
        *,
        actor_name: str,
        scene_name: str = DEFAULT_SCENE_NAME,
    ) -> str:
        try:
            scene = _resolve_scene(scene_manager, scene_name)
            if scene is None:
                return build_error_result(
                    error_message="没有已加载的场景，请先打开或创建一个场景"
                ).to_envelope(interface_type="scene")

            actor = scene.find_actor(actor_name)
            if actor is None:
                available = [a.name for a in scene.get_actors()]
                return build_error_result(
                    error_message=f"未找到名为 '{actor_name}' 的模型。"
                                  f"当前场景中的模型: {available}"
                ).to_envelope(interface_type="scene")

            scene.remove_actor(actor)

            result_data = {
                "status": "success",
                "removed_actor": actor_name,
                "scene": getattr(scene, "name", scene_name),
                "remaining_actors": [a.name for a in scene.get_actors()],
            }
            part = build_part(
                content_type="text",
                content_text=json.dumps(result_data, ensure_ascii=False),
            )
            return build_success_result(parts=[part]).to_envelope(
                interface_type="scene"
            )

        except Exception as e:
            return build_error_result(
                error_message=f"模型删除失败: {e}"
            ).to_envelope(interface_type="scene")

    return StructuredTool(
        name="remove_model", 
        description="从当前场景中删除指定的模型（Actor）。"
                    "需要提供模型名称，支持模糊匹配（忽略引号、扩展名）。",
        args_schema=RemoveModelInput,
        func=_remove_model,
    )


# ---------------------------------------------------------------------------
# Public Loader
# ---------------------------------------------------------------------------

def load_model_import_tools() -> List[StructuredTool]:
    from CoronaCore.core.managers import scene_manager
    return [
        _build_import_model_tool(scene_manager),
        _build_remove_model_tool(scene_manager),
    ]


__all__ = ["load_model_import_tools"]
