import os
import json
import logging
import datetime
import time
from typing import Any, Dict

from config.paths_config import get_project_screenshots_dir
from CoronaCore.core.managers import scene_manager
from ai_workflow.streaming import stream_output_node
from .formatters import NO_OUTPUT
from .helpers import get_tool

logger = logging.getLogger(__name__)

def _resolve_active_scene():
    """健壮的场景获取逻辑"""
    scene = scene_manager.get("")
    if scene is not None:
        return scene
    routes = scene_manager.list_all()
    if routes:
        return scene_manager.get(routes[0])
    return None

def _resolve_model_file(model_path: str) -> str:
    """【核心新增】：解析相对路径并找出真实的 3D 模型文件 (.glb/.obj 等)"""
    from CoronaCore.core.corona_editor import CoronaEditor
    CoronaEngine = CoronaEditor.CoronaEngine
    
    # 1. 相对路径转绝对路径
    if os.path.isabs(model_path):
        resolved_path = model_path
    else:
        project_path = getattr(CoronaEngine, "active_project_path", "")
        if project_path:
            resolved_path = os.path.join(project_path, model_path)
        else:
            resolved_path = model_path

    # 2. 如果是文件，直接返回
    SUPPORTED_EXTS = {".obj", ".dae", ".glb", ".gltf", ".fbx"}
    if os.path.isfile(resolved_path):
        if any(resolved_path.lower().endswith(ext) for ext in SUPPORTED_EXTS):
            return resolved_path
        return ""

    # 3. 如果是目录，寻找第一个支持的模型文件
    if os.path.isdir(resolved_path):
        for ext in SUPPORTED_EXTS:
            for f in sorted(os.listdir(resolved_path)):
                if f.lower().endswith(ext):
                    return os.path.join(resolved_path, f)
    return ""


@stream_output_node("integrated", NO_OUTPUT)
def six_view_capture_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    model_results = state.get("model_results", [])
    if not model_results:
        return {"six_view_images": {}}

    multiview_tool = get_tool("camera_multiview_capture")
    if not multiview_tool:
        return {"six_view_images": {}}

    scene = _resolve_active_scene()
    if scene is None:
        logger.warning("[Workflow] 未加载任何场景，无法执行截图")
        return {"six_view_images": {}}

    all_saved_views = {}

    for result in model_results:
        if result.get("error"):
            continue
            
        actor_name = result.get("object_id") or result.get("item_name")
        raw_model_path = result.get("model_path")
        
        if not actor_name or not raw_model_path:
            continue

        # 解析真正的 .glb/.obj 文件路径
        final_model_path = _resolve_model_file(raw_model_path)
        
        # 确定截图的输出目录（放在原路径同级或子文件夹）
        if os.path.isabs(raw_model_path):
            model_dir = raw_model_path if os.path.isdir(raw_model_path) else os.path.dirname(raw_model_path)
        else:
            from CoronaCore.core.corona_editor import CoronaEditor
            project_path = getattr(CoronaEditor.CoronaEngine, "active_project_path", str(get_project_screenshots_dir()))
            model_dir = os.path.join(project_path, raw_model_path)
            model_dir = model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir)
            
        output_dir = model_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"[Workflow] 正在准备为 {actor_name} 生成环绕视图...")

        actor = scene.find_actor(actor_name)
        is_temp_loaded = False

        # 【逻辑修正】：使用真实找到的 final_model_path 判断
        if not actor and final_model_path and os.path.exists(final_model_path):
            logger.info(f"[Workflow] 场景中未找到 {actor_name}，正在从硬盘加载真实文件: {final_model_path}")
            try:
                from CoronaCore.core.entities.actor import Actor
                # 必须传入具体的 .glb 等文件
                actor = Actor(
                    name=actor_name,
                    route=final_model_path,
                    actor_type="mesh",
                    parent_scene=scene,
                )
                scene.add_actor(actor)
                is_temp_loaded = True
                
                # 给予引擎充分的时间加载网格和材质
                time.sleep(1.0) 
            except Exception as e:
                logger.error(f"[Workflow] 临时加载模型失败: {e}")
                continue
        elif not final_model_path:
            logger.warning(f"[Workflow] 无法在 {raw_model_path} 及其子目录下找到受支持的 3D 模型文件 (.glb/.obj)，跳过。")

        if not scene.find_actor(actor_name):
            logger.warning(f"[Workflow] 无法加载 {actor_name}，跳过截图。")
            continue

        try:
            # 获取场景中的主相机
            camera = scene.find_camera(None)
            if not camera:
                logger.error(f"[Workflow] 场景中没有可用相机，跳过 {actor_name} 截图")
                continue

            # --- 1. 精准计算物体的中心点和观察距离 ---
            import math
            aabb = actor._geometry.get_aabb()
            actor_pos = actor.get_position()
            actor_scale = actor.get_scale()
            
            model_center = [
                (aabb[0] + aabb[3]) / 2.0,
                (aabb[1] + aabb[4]) / 2.0,
                (aabb[2] + aabb[5]) / 2.0,
            ]
            center = [
                actor_pos[0] + model_center[0] * actor_scale[0],
                actor_pos[1] + model_center[1] * actor_scale[1],
                actor_pos[2] + model_center[2] * actor_scale[2],
            ]
            dx = (aabb[3] - aabb[0]) * actor_scale[0]
            dy = (aabb[4] - aabb[1]) * actor_scale[1]
            dz = (aabb[5] - aabb[2]) * actor_scale[2]
            # 乘以 1.5 倍对角线长度，留出画面安全边距
            distance = max(math.sqrt(dx*dx + dy*dy + dz*dz) * 1.5, 1.0)
            fov = camera.get_fov()

            # --- 2. 定义标准六视图的精确三维角度：(仰角, 偏航角) ---
            view_configs = {
                "front":  (0.0, 0.0),      # 前
                "back":   (0.0, 180.0),    # 后
                "left":   (0.0, 270.0),    # 左
                "right":  (0.0, 90.0),     # 右
                "top":    (90.0, 0.0),     # 上
                "bottom": (-90.0, 0.0)     # 下
            }

            view_dict = {}

            # --- 3. 循环将相机摆放到 6 个精确位置并截图 ---
            for view_name, (elev_deg, az_deg) in view_configs.items():
                elev_rad, az_rad = math.radians(elev_deg), math.radians(az_deg)
                cos_elev, sin_elev = math.cos(elev_rad), math.sin(elev_rad)
                cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)

                # 计算相机的球面坐标偏移
                offset_x = distance * cos_elev * sin_az
                offset_y = distance * sin_elev
                offset_z = distance * cos_elev * cos_az

                position = [center[0] + offset_x, center[1] + offset_y, center[2] + offset_z]
                
                # 计算相机朝向 (死死盯住物体中心)
                fwd = [center[0] - position[0], center[1] - position[1], center[2] - position[2]]
                fwd_len = math.sqrt(sum(f*f for f in fwd))
                fwd = [f / fwd_len for f in fwd] if fwd_len > 1e-6 else [0.0, 0.0, -1.0]

                # 【防万向锁】处理正上方和正下方时，相机的 Up 向量必须重置
                if elev_deg >= 89.0:
                    up = [0.0, 0.0, 1.0]
                elif elev_deg <= -89.0:
                    up = [0.0, 0.0, -1.0]
                else:
                    up = [0.0, 1.0, 0.0]

                # 移动相机并等待渲染管线刷新 (极其重要)
                camera.set(position, fwd, up, fov)
                time.sleep(0.15) 

                # 截图并保存字典（必须用同步版本，否则清理模型时渲染线程仍在写入会导致 SIGABRT）
                filepath = os.path.join(output_dir, f"{view_name}.png")
                camera.save_screenshot_sync(filepath)
                view_dict[view_name] = filepath

            # 记录这 6 张图的结果
            all_saved_views[actor_name] = view_dict
            logger.info(f"[Workflow] {actor_name} 标准六视图 (前后左右上下) 生成成功: 6 张")
            
            result["six_views_dict"] = view_dict

        except Exception as e:
            logger.error(f"[Workflow] 截图执行崩溃: {e}", exc_info=True)
            
        finally:
            if is_temp_loaded and actor:
                logger.info(f"[Workflow] 截图完毕，正在清理临时加载的模型: {actor_name}")
                try:
                    scene.remove_actor(actor)
                except Exception as e:
                    logger.error(f"[Workflow] 清理模型失败: {e}")

    return {"model_results": model_results, "six_view_images": all_saved_views}