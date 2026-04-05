from __future__ import annotations

import json
import math
from typing import Literal, TYPE_CHECKING, Tuple, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_modules.mcp.configs.prompts import (
    SCENE_QUERY_PROMPTS,
    SCENE_TRANSFORM_PROMPTS,
)
from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)
DEFAULT_SCENE_NAME = ""


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


class SceneQueryInput(BaseModel):
    scene_name: str = Field(default=DEFAULT_SCENE_NAME, description="要查询的场景名称")
    query: Literal["list_models", "get_model_by_name"] = Field(description="查询类型")
    name: str | None = Field(
        default=None, description=SCENE_QUERY_PROMPTS.fields["model_name"]
    )


class TransformModelInput(BaseModel):
    scene_name: str = Field(default=DEFAULT_SCENE_NAME, description="目标场景名称")
    model_name: str = Field(description="需要变换的模型名称")
    operation: Literal["scale", "move", "rotate"] = Field(
        default="scale", description=SCENE_TRANSFORM_PROMPTS.fields["transform_type"]
    )
    scale_factor: float | None = Field(
        default=None,
        description=SCENE_TRANSFORM_PROMPTS.fields["value"],
    )
    vector: Tuple[float, float, float] | None = Field(
        default=None,
        description=SCENE_TRANSFORM_PROMPTS.fields["axis"],
    )


class SceneActorsInput(BaseModel):
    scene_name: str = Field(
        default=DEFAULT_SCENE_NAME,
        description="目标场景名称，为空则使用当前场景",
    )
    actor_name: str | None = Field(
        default=None,
        description="指定物体名称以获取详细信息；为空则列出所有物体",
    )


class SceneListInput(BaseModel):
    """列出所有已加载场景，无需参数"""
    pass


def _build_scene_query_tool(scene_manager) -> StructuredTool:
    def _query_scene(
        *,
        scene_name: str = DEFAULT_SCENE_NAME,
        query: Literal["list_models", "get_model_by_name"],
        name: str | None = None,
    ) -> str:
        try:
            data = SceneQueryInput(scene_name=scene_name, query=query, name=name)
            scene = _resolve_scene(scene_manager, data.scene_name)

            result_data = {}
            if scene is None:
                result_data = {"scene": data.scene_name, "actors": []}
            elif data.query == "list_models":
                actors = [actor.name for actor in scene.get_actors()]
                result_data = {"scene": data.scene_name, "actors": actors}
            elif data.query == "get_model_by_name":
                actor = scene.find_actor(data.name or "")
                if actor is None:
                    result_data = {
                        "scene": data.scene_name,
                        "actor": None,
                        "found": False,
                    }
                else:
                    result_data = {
                        "scene": data.scene_name,
                        "actor": actor.name,
                        "model_path": getattr(actor, "model_path", ""),
                        "found": True,
                    }
            else:
                return build_error_result(
                    error_message=f"Unsupported query type: {data.query}"
                ).to_envelope(interface_type="scene")

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
        name="scene_query",
        description=SCENE_QUERY_PROMPTS.tool_description,
        args_schema=SceneQueryInput,
        func=_query_scene,
    )


def _build_transform_tool(scene_manager) -> StructuredTool:
    def _transform_model(
        *,
        scene_name: str = DEFAULT_SCENE_NAME,
        model_name: str,
        operation: Literal["scale", "move", "rotate"] = "scale",
        scale_factor: float | None = None,
        vector: Tuple[float, float, float] | None = None,
    ) -> str:
        try:
            data = TransformModelInput(
                scene_name=scene_name,
                model_name=model_name,
                operation=operation,
                scale_factor=scale_factor,
                vector=vector,
            )
            scene = _resolve_scene(scene_manager, data.scene_name)
            if scene is None:
                return build_error_result(
                    error_message="No scene loaded"
                ).to_envelope(interface_type="scene")

            actor = scene.find_actor(data.model_name)
            if actor is None:
                return build_error_result(
                    error_message=f"Actor '{data.model_name}' not found"
                ).to_envelope(interface_type="scene")

            op = data.operation.lower()
            if op == "scale":
                if data.scale_factor is not None:
                    v = [data.scale_factor] * 3
                elif data.vector is not None:
                    v = list(data.vector)
                else:
                    raise ValueError("scale 操作需要提供 scale_factor 或 vector")
                actor.set_scale(v)
            elif op == "move":
                if data.vector is None:
                    raise ValueError("move 操作需要提供 vector")
                actor.move(list(data.vector))
            elif op == "rotate":
                if data.vector is None:
                    raise ValueError("rotate 操作需要提供 vector")
                actor.rotate(list(data.vector))
            else:
                return build_error_result(
                    error_message=f"Unsupported operation '{data.operation}'"
                ).to_envelope(interface_type="scene")

            payload = {
                "actor": actor.name,
                "operation": op,
                "position": list(actor.get_position()),
                "rotation": list(actor.get_rotation()),
                "scale": list(actor.get_scale()),
            }
            part = build_part(
                content_type="text",
                content_text=json.dumps(payload, ensure_ascii=False),
            )
            return build_success_result(parts=[part]).to_envelope(
                interface_type="scene"
            )
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="scene"
            )

    return StructuredTool(
        name="transform_model",
        description=SCENE_TRANSFORM_PROMPTS.tool_description,
        args_schema=TransformModelInput,
        func=_transform_model,
    )


def _build_scene_actors_tool(scene_manager) -> StructuredTool:
    """构建场景物体查询工具"""

    def _scene_actors(
        *,
        scene_name: str = DEFAULT_SCENE_NAME,
        actor_name: str | None = None,
    ) -> str:
        try:
            scene = _resolve_scene(scene_manager, scene_name)
            if scene is None:
                return build_error_result(
                    error_message="No scene loaded"
                ).to_envelope(interface_type="scene")

            # 查询单个物体详情
            if actor_name:
                actor = scene.find_actor(actor_name)
                if actor is None:
                    return build_error_result(
                        error_message=f"Actor '{actor_name}' not found"
                    ).to_envelope(interface_type="scene")

                info = {
                    "name": actor.name,
                    "actor_type": getattr(actor, "actor_type", "unknown"),
                    "model_path": getattr(actor, "model_path", ""),
                }

                # Transform
                try:
                    info["position"] = list(actor.get_position())
                    info["rotation"] = list(actor.get_rotation())
                    info["scale"] = list(actor.get_scale())
                except Exception:
                    pass

                # AABB
                geom = getattr(actor, "_geometry", None)
                if geom is not None:
                    try:
                        aabb = geom.get_aabb()
                        info["aabb"] = {
                            "min": [aabb[0], aabb[1], aabb[2]],
                            "max": [aabb[3], aabb[4], aabb[5]],
                        }
                        dx = aabb[3] - aabb[0]
                        dy = aabb[4] - aabb[1]
                        dz = aabb[5] - aabb[2]
                        info["size"] = [dx, dy, dz]
                    except Exception:
                        pass

                # Physics
                mechanics = getattr(actor, "_mechanics", None)
                if mechanics is not None:
                    try:
                        info["physics"] = {
                            "mass": mechanics.get_mass(),
                            "restitution": mechanics.get_restitution(),
                            "damping": mechanics.get_damping(),
                        }
                    except Exception:
                        pass

                part = build_part(
                    content_type="text",
                    content_text=json.dumps(info, ensure_ascii=False),
                )
                return build_success_result(parts=[part]).to_envelope(
                    interface_type="scene"
                )

            # 列出所有物体
            actors_list = []
            for actor in scene.get_actors():
                entry = {
                    "name": actor.name,
                    "actor_type": getattr(actor, "actor_type", "unknown"),
                }
                try:
                    entry["position"] = list(actor.get_position())
                except Exception:
                    pass
                actors_list.append(entry)

            result_data = {
                "scene": getattr(scene, "name", scene_name),
                "count": len(actors_list),
                "actors": actors_list,
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
        name="scene_get_actors",
        description=(
            "获取场景中的物体信息。不传 actor_name 时列出场景中所有物体及其位置；"
            "传入 actor_name 时返回该物体的详细信息，包括位置、旋转、缩放、包围盒、物理属性等。"
            "坐标系：X正为右，Y正为上，Z正为朝屏幕里侧（左手坐标系）。"
        ),
        args_schema=SceneActorsInput,
        func=_scene_actors,
    )


def _build_scene_list_tool(scene_manager) -> StructuredTool:
    """构建列出所有已加载场景的工具"""

    def _scene_list() -> str:
        try:
            routes = scene_manager.list_all()
            scenes_info = []
            for route in routes:
                scene = scene_manager.get(route)
                entry = {"route": route}
                if scene is not None:
                    entry["name"] = getattr(scene, "name", route)
                    try:
                        entry["actor_count"] = len(scene.get_actors())
                    except Exception:
                        pass
                    try:
                        entry["camera_count"] = len(scene.get_cameras())
                    except Exception:
                        pass
                scenes_info.append(entry)

            result_data = {
                "count": len(scenes_info),
                "scenes": scenes_info,
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
        name="scene_list",
        description="列出所有已加载的场景，返回每个场景的路由名称、显示名称、物体数量和摄像头数量。",
        args_schema=SceneListInput,
        func=_scene_list,
    )


def load_scene_tools() -> List[StructuredTool]:
    from CoronaCore.core.managers import scene_manager
    return [
        _build_scene_list_tool(scene_manager),
        _build_scene_actors_tool(scene_manager),
        _build_scene_query_tool(scene_manager),
        _build_transform_tool(scene_manager),
    ]


__all__ = ["load_scene_tools"]
