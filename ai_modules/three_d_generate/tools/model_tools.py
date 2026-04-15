from __future__ import annotations

import logging
import os
import time
import threading
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ai_config.ai_config import AIConfig
from ai_media_resource import get_media_registry
from ai_tools.context import get_current_session
from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)
from ai_config.paths_config import get_project_models_dir, _get_active_project_path
from ai_modules.three_d_generate.tools.client_3d import Rodin3DClient
from ai_modules.three_d_generate.tools.client_hunyuan3d import Hunyuan3DClient

import re
import urllib.parse

_WIN_INVALID = r'[<>:"/\\|?*\x00-\x1F]'
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mesh 下载完成 Event 注册表
# ---------------------------------------------------------------------------
# 后台线程下载 mesh 时，工作流可通过 wait_for_mesh_ready() 阻塞等待。
# Event 在 bg_thread.start() 前注册，在 _download_rest_files_async 的
# finally 中 set，保证无竞争窗口且不会永久阻塞。

_MESH_READY_EVENTS: Dict[str, threading.Event] = {}
_MESH_EVENTS_LOCK = threading.Lock()


def _register_mesh_event(object_id: str) -> None:
    """为指定 object_id 创建 mesh 下载完成 Event（必须在后台线程启动前调用）。"""
    with _MESH_EVENTS_LOCK:
        _MESH_READY_EVENTS[object_id] = threading.Event()


def _signal_mesh_ready(object_id: str) -> None:
    """通知 mesh 下载已完成（成功或失败），唤醒所有等待者。"""
    with _MESH_EVENTS_LOCK:
        event = _MESH_READY_EVENTS.get(object_id)
    if event is not None:
        event.set()


def wait_for_mesh_ready(object_id: str) -> bool:
    """阻塞等待指定 object_id 的 mesh 下载完成。

    若 object_id 不在注册表（表示无后台下载任务），立即返回 True。
    等待完成后自动清理注册表条目。

    Returns:
        True 表示等待完成（或无需等待）。
    """
    with _MESH_EVENTS_LOCK:
        event = _MESH_READY_EVENTS.get(object_id)

    if event is None:
        return True

    event.wait()

    with _MESH_EVENTS_LOCK:
        _MESH_READY_EVENTS.pop(object_id, None)

    return True


def _sanitize_name(s: str, allow_spaces: bool = False) -> str:
    """
    通用文件/目录名清理函数
    - allow_spaces=False: 用于目录名（删除空格）
    - allow_spaces=True: 用于文件名（保留单个空格）
    """
    s = (s or "").strip().replace("\\", "_").replace("/", "_")
    if not allow_spaces:
        # 目录名模式：删除所有空格，归一化连字符
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"\s+", "", s)
    else:
        # 文件名模式：多个空格变单个
        s = re.sub(r"\s+", " ", s).strip()
    # 替换非法字符
    s = re.sub(_WIN_INVALID, "_", s)
    return s or "task"


def _safe_dirname(s: str) -> str:
    """目录名清理（删除所有空格）"""
    return _sanitize_name(s, allow_spaces=False)


def _safe_filename(name: str) -> str:
    """文件名清理（保留单个空格）"""
    return _sanitize_name(name, allow_spaces=True)


def _filename_from_url(url: str) -> str:
    u = urllib.parse.urlparse(url)
    base = (u.path or "").rstrip("/").split("/")[-1]
    base = _safe_filename(base)
    return base if "." in base else (base + ".bin")


def _download_url_to_dir(
    url: str,
    out_dir: str,
    timeout: float = 120.0,
    preferred_filename: Optional[str] = None,
) -> str:
    """下载 url 到 out_dir，返回保存后的绝对路径"""
    os.makedirs(out_dir, exist_ok=True)

    filename = (
        _safe_filename(preferred_filename)
        if preferred_filename
        else _filename_from_url(url)
    )
    dest = os.path.join(out_dir, filename)

    # 避免覆盖：若已存在则追加序号
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        base, ext = os.path.splitext(filename)
        i = 1
        while True:
            cand = os.path.join(out_dir, f"{base}_{i}{ext}")
            if not (os.path.exists(cand) and os.path.getsize(cand) > 0):
                dest = cand
                break
            i += 1

    tmp_dest = dest + ".tmp"
    if os.path.exists(tmp_dest):
        try:
            os.remove(tmp_dest)
        except Exception:
            pass

    # 增强下载鲁棒性，避免 HTTP 连接中途断开导致残留不完整文件
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=httpx.Timeout(timeout, connect=30.0, read=max(timeout, 300.0), write=max(timeout, 300.0)),
    ) as r:
        r.raise_for_status()
        content_length = 0
        try:
            content_length = int(r.headers.get("content-length", "0"))
        except (TypeError, ValueError):
            content_length = 0

        bytes_written = 0
        with open(tmp_dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
            f.flush()
            os.fsync(f.fileno())

    if content_length > 0 and bytes_written < content_length:
        os.remove(tmp_dest)
        raise IOError(
            f"文件下载不完整: {url}, 期待 {content_length} bytes, 实际 {bytes_written} bytes"
        )

    os.replace(tmp_dest, dest)
    return dest


def _to_repo_relative_path(absolute_path: str, repo_root: Path) -> str:
    """将绝对路径转换为仓库相对路径，统一使用 POSIX 分隔符。"""
    absolute = Path(absolute_path).resolve()
    root = repo_root.resolve()
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError as e:
        raise RuntimeError(f"模型文件不在仓库目录内: {absolute}") from e


class RodinGenerate3DInput(BaseModel):
    """
    Rodin 3D 生成（显式 mode）
    """

    mode: str = Field(default="image_to_3d", description="image_to_3d / text_to_3d")

    images: Optional[List[str]] = Field(
        default=None,
        description="图片输入（URL 或本地路径）。mode=image_to_3d 时必填",
    )

    prompt: Optional[str] = Field(
        default=None,
        description="文本提示词。mode=text_to_3d 时必填",
    )

    object_id: Optional[str] = Field(
        default=None,
        description="模型文件名使用的对象标识（不含扩展名）",
    )

    condition_mode: str = Field(default="concat")
    tier: str = Field(default="Regular")
    quality: Optional[str] = None
    seed: Optional[int] = None
    geometry_file_format: str = Field(default="glb")
    material: Optional[str] = None
    addons: Optional[str] = None

    download_dir: Optional[str] = Field(
        default=None,
        description="已保留但不生效：3D 模型固定保存到项目 assets/model 目录",
    )


def load_3d_tools(config: AIConfig) -> List[StructuredTool]:
    threed_config = config.rodin3d

    provider_name = (getattr(threed_config, "provider", "") or "").strip()
    provider_cfg = (config.providers or {}).get(provider_name) if provider_name else None

    base_url = (
        (getattr(threed_config, "base_url", "") or "").strip()
        or (getattr(provider_cfg, "base_url", "") or "").strip()
    )
    api_key = (
        (getattr(threed_config, "api_key", "") or "").strip()
        or (getattr(provider_cfg, "api_key", "") or "").strip()
    )
    if not base_url:
        raise RuntimeError("Rodin base_url 缺失：请在 settings.rodin_3d.base_url 配置")
    if not api_key:
        raise RuntimeError("Rodin api_key 缺失：请在 settings.rodin_3d.api_key 配置")

    client = Rodin3DClient(
        base_url=base_url,
        api_key=api_key,
        timeout=float(threed_config.request_timeout),
    )

    generate_path = threed_config.generate_path
    status_path = threed_config.status_path
    download_path = threed_config.download_path
    poll_interval = threed_config.poll_interval
    poll_timeout = threed_config.poll_timeout

    media_registry = get_media_registry()

    # ==================== 后台异步函数 ====================
    def _download_rest_files_async(
        object_dir: Path,
        repo_root: Path,
        downloads: List[Dict[str, str]],
        object_dir_name: str,
        model_object_id: str,
        mode: str,
        geometry_file_format: str,
        tier: str,
        quality: Optional[str],
        batch_tag: str,
        session_id: str,
        registry,
    ):
        """后台异步下载 mesh 和其他非 preview 文件，并注册到资源管理器"""
        logger = logging.getLogger(__name__)
        try:
            mesh_count = 0
            for it in downloads:
                url = str(it.get("url", "")).strip()
                if not url:
                    continue

                ext = os.path.splitext(_filename_from_url(url))[1].lower() or ".bin"

                # 只下 mesh 等，preview 已在前面处理过
                if ext in {".webp", ".png", ".jpg", ".jpeg"}:
                    continue

                if ext in {".glb", ".gltf", ".obj", ".fbx"}:
                    typ = "mesh"
                    mesh_count += 1
                    preferred = f"{model_object_id}{ext}"
                    output_type = "file"
                else:
                    typ = ext
                    preferred = f"{typ}_{batch_tag}{ext}"
                    output_type = "file"

                try:
                    local_path = _download_url_to_dir(
                        url,
                        str(object_dir),
                        timeout=float(threed_config.request_timeout),
                        preferred_filename=preferred,
                    )
                    relative_path = _to_repo_relative_path(local_path, repo_root)

                    # ✅ 关键：注册到资源管理器，使前端可以通过 fileid:// 访问
                    file_id = registry.register(
                        session_id=session_id,
                        content_url=str(Path(local_path).resolve()),
                        resource_type=output_type,
                        content_text=relative_path,
                        parameter={
                            "additional_type": ["rodin_3d"],
                            "object_id": object_dir_name,
                            "mode": mode,
                            "geometry_file_format": geometry_file_format,
                            "tier": tier,
                            "quality": quality,
                            "name": it.get("name"),
                            "short_id": model_object_id if typ == "mesh" else typ,
                        },
                    )

                    logger.debug(f"后台下载+注册完成: {url} -> {relative_path} (file_id={file_id})")
                except Exception as e:
                    logger.warning(f"后台下载/注册失败: {url}, err={e}")

            logger.info(f"Rodin 3D 后台异步下载完成: {object_dir}, mesh count={mesh_count}")

        except Exception as e:
            logger.error(f"Rodin 3D 后台下载异常: {e}")
        finally:
            _signal_mesh_ready(object_dir_name)

    # ==================== 主工具函数 ====================
    def _rodin_generate_3d(
        mode: str = "image_to_3d",
        images: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        object_id: Optional[str] = None,
        condition_mode: str = "concat",
        tier: str = "Regular",
        quality: Optional[str] = None,
        seed: Optional[int] = None,
        geometry_file_format: str = "glb",
        material: Optional[str] = None,
        addons: Optional[str] = None,
        download_dir: Optional[str] = None,
    ) -> str:
        logger = logging.getLogger(__name__)
        try:
            # 每次调用时动态获取项目路径，确保跟随当前活跃项目
            repo_root = _get_active_project_path().resolve()
            assets_model_root = get_project_models_dir().resolve()

            mode = (mode or "").strip()
            requested_object_id = str(object_id or "").strip()
            mesh_object_id = _safe_filename(requested_object_id) if requested_object_id else "base"
            if requested_object_id and requested_object_id != mesh_object_id:
                logger.warning(
                    "object_id 包含非法文件名字符，已规范化: raw=%s, sanitized=%s",
                    requested_object_id,
                    mesh_object_id,
                )

            # 解析 fileid:// -> http(s) url
            image_list: List[str] = []
            for image in images or []:
                if isinstance(image, str) and image.startswith("fileid://"):
                    file_id = image[9:].strip()
                    # 阻塞等待异步任务完成，避免读取到空 content_url。
                    image_list.append(media_registry.resolve(file_id))
                else:
                    image_list.append(image)

            prompt = prompt.strip() if isinstance(prompt, str) else None
            if mode not in {"image_to_3d", "text_to_3d"}:
                raise ValueError("mode 必须是 image_to_3d 或 text_to_3d")

            if mode == "image_to_3d" and not image_list:
                raise ValueError("image_to_3d 模式必须提供 images")
            if mode == "text_to_3d" and not prompt:
                raise ValueError("text_to_3d 模式必须提供 prompt")

            form_fields: Dict[str, Any] = {
                "prompt": prompt,
                "condition_mode": condition_mode,
                "tier": tier,
                "quality": quality,
                "seed": seed,
                "geometry_file_format": geometry_file_format,
                "material": material,
                "addons": addons,
            }

            result = client.run_to_download_urls(
                generate_path=generate_path,
                status_path=status_path,
                download_path=download_path,
                images=image_list if mode == "image_to_3d" else None,
                form_fields=form_fields,
                poll_interval=poll_interval,
                poll_timeout=poll_timeout,
            )

            logger.info(
                "Rodin 3D done task_uuid=%s downloads=%s",
                result.get("task_uuid"),
                len(result.get("downloads") or []),
            )

            # -----------------------------
            # 下载到本地
            # 固定保存到 <repo_root>/assets/model
            # -----------------------------
            cfg_download_dir = getattr(threed_config, "download_dir", None)
            env_download_dir = os.environ.get("RODIN_3D_DOWNLOAD_DIR")
            if download_dir or cfg_download_dir or env_download_dir:
                logger.warning(
                    "3D 保存目录已固定为 assets/model，忽略 download_dir 配置: arg=%s, cfg=%s, env=%s",
                    download_dir,
                    cfg_download_dir,
                    env_download_dir,
                )

            # ✅ 不用 task_uuid 建目录，改用 batch 目录（时间戳）
            # 修改：根据第一个下载 URL 提取语义化目录名，优先使用有意义名字
            downloads = result.get("downloads") or []
            if not downloads:
                raise RuntimeError("Rodin 未返回任何可下载文件（downloads 为空）")

            first_url = None
            for it in downloads:
                if not isinstance(it, dict):
                    continue
                candidate = str(it.get("url") or it.get("content_url") or "").strip()
                if candidate:
                    first_url = candidate
                    break

            if requested_object_id:
                # 优先使用语义化 object_id 命名目录，保证 auto_scan 入库时 id 与工作流一致
                object_dir_name = _safe_dirname(requested_object_id)
            elif first_url:
                candidate_name = os.path.splitext(_filename_from_url(first_url))[0]
                object_dir_name = _safe_dirname(candidate_name)
            else:
                object_dir_name = "模型"

            # 避免目录冲突：存在则加后缀
            original_dir_name = object_dir_name
            suffix_idx = 1
            while (assets_model_root / object_dir_name).exists():
                object_dir_name = f"{original_dir_name}_{suffix_idx}"
                suffix_idx += 1

            object_dir = assets_model_root / object_dir_name
            object_dir.mkdir(parents=True, exist_ok=True)
            batch_tag = str(int(time.time()))

            # ✅ 分离 preview 和 mesh 下载：preview 同步、mesh 后台
            registry = get_media_registry()
            session_id = get_current_session()

            preview_items = []
            rest_items = []

            for it in downloads:
                url = str(it.get("url", "")).strip()
                if not url:
                    continue
                ext = os.path.splitext(_filename_from_url(url))[1].lower() or ".bin"

                if ext in {".webp", ".png", ".jpg", ".jpeg"}:
                    preview_items.append(it)
                else:
                    rest_items.append(it)

            # ----- 第一阶段：同步下载 preview -----
            preview_parts = []
            preview_count = 0

            for it in preview_items:
                url = str(it.get("url", "")).strip()
                if not url:
                    continue

                ext = os.path.splitext(_filename_from_url(url))[1].lower() or ".bin"
                preview_count += 1
                preferred = f"preview_{preview_count:04d}{ext}"
                output_type = "image"

                try:
                    local_path = _download_url_to_dir(
                        url,
                        str(object_dir),
                        timeout=float(threed_config.request_timeout),
                        preferred_filename=preferred,
                    )
                    relative_path = _to_repo_relative_path(local_path, repo_root)

                    # 注册 preview 资源
                    file_id = registry.register(
                        session_id=session_id,
                        content_url=str(Path(local_path).resolve()),
                        resource_type=output_type,
                        content_text=relative_path,
                        parameter={
                            "additional_type": ["rodin_3d"],
                            "object_id": object_dir_name,
                            "mode": mode,
                            "geometry_file_format": geometry_file_format,
                            "tier": tier,
                            "quality": quality,
                            "name": it.get("name"),
                            "short_id": str(preview_count),
                        },
                    )

                    preview_parts.append(
                        build_part(
                            content_type=output_type,
                            content_text=relative_path,
                            file_id=file_id,
                        )
                    )

                    logger.debug(f"预览图下载完成: {url} -> {relative_path}")

                except Exception as e:
                    logger.warning(f"预览图下载失败: {url}, err={e}")

            # ---- 第二阶段：立即返回（只包含 preview + model_folder 信息）----
            # 允许 preview 为空但有 mesh 的情况（后台会处理）
            if not preview_parts and not rest_items:
                raise RuntimeError("未能获取任何下载资源（既无预览图也无模型文件）")

            model_folder_relative = _to_repo_relative_path(str(object_dir), repo_root)

            # 启动后台线程继续下载 mesh 等，并注册到资源管理器
            if rest_items:
                # 在线程启动前注册 Event，保证 wait 方不会错过信号
                _register_mesh_event(object_dir_name)
                bg_thread = threading.Thread(
                    target=_download_rest_files_async,
                    args=(
                        object_dir,
                        repo_root,
                        rest_items,
                        object_dir_name,
                        mesh_object_id,
                        mode,
                        geometry_file_format,
                        tier,
                        quality,
                        batch_tag,
                        session_id,
                        registry,
                    ),
                    daemon=True,
                )
                bg_thread.start()
                logger.info(
                    f"Rodin 3D 后台异步任务已启动: {object_dir}, 将下载并注册 {len(rest_items)} 个资源"
                )

            # ---- 构造最终返回 ----
            return build_success_result(
                parts=preview_parts,
                metadata={
                    "model_folder": model_folder_relative,
                    "folder_object_id": object_dir_name,
                    "mesh_object_id": mesh_object_id,
                    "requested_object_id": requested_object_id,
                    "task_uuid": result.get("task_uuid"),
                    "preview_count": len(preview_parts),
                    "has_mesh_pending": len(rest_items) > 0,
                },
            ).to_envelope(interface_type="media")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="media"
            )

    return [
        StructuredTool(
            name="rodin_generate_3d",
            description="调用 Rodin API 生成 3D（image_to_3d / text_to_3d）",
            func=_rodin_generate_3d,
            args_schema=RodinGenerate3DInput,
        )
    ]


# ===========================================================================
# 混元3D 工具
# ===========================================================================

class Hunyuan3DGenerate3DInput(BaseModel):
    """混元3D 生成输入"""

    mode: str = Field(default="image_to_3d", description="image_to_3d / text_to_3d")

    images: Optional[List[str]] = Field(
        default=None,
        description="图片输入（URL 或本地路径）。mode=image_to_3d 时必填",
    )

    prompt: Optional[str] = Field(
        default=None,
        description="文本提示词。mode=text_to_3d 时必填",
    )

    result_format: str = Field(default="GLB", description="输出格式：GLB, OBJ, STL, USDZ, FBX")
    enable_pbr: bool = Field(default=False, description="是否开启 PBR 材质")
    generate_type: str = Field(default="Normal", description="Normal, LowPoly, Geometry, Sketch")
    model_version: str = Field(default="3.0", description="模型版本：3.0, 3.1")
    face_count: Optional[int] = Field(default=None, description="面数，默认 500000")

    download_dir: Optional[str] = Field(
        default=None,
        description="已保留但不生效：3D 模型固定保存到项目 assets/model 目录",
    )


def load_hunyuan3d_tools(config: AIConfig) -> List[StructuredTool]:
    hunyuan_config = config.hunyuan3d

    if not getattr(hunyuan_config, 'enable', False):
        logger.info("混元3D 已禁用 (enable=False)，跳过工具加载")
        return []

    api_key = (hunyuan_config.api_key or "").strip()
    if not api_key:
        raise RuntimeError("混元3D api_key 缺失：请在 settings.hunyuan3d.api_key 配置")

    client = Hunyuan3DClient(
        api_key=api_key,
        region=hunyuan_config.region,
        endpoint=hunyuan_config.endpoint,
        timeout=float(hunyuan_config.request_timeout),
        version=hunyuan_config.version,
    )

    poll_interval = hunyuan_config.poll_interval
    poll_timeout = hunyuan_config.poll_timeout
    default_result_format = hunyuan_config.result_format
    default_enable_pbr = hunyuan_config.enable_pbr
    default_generate_type = hunyuan_config.generate_type
    default_model = hunyuan_config.model
    default_face_count = hunyuan_config.face_count

    media_registry = get_media_registry()

    # ==================== 后台异步函数 ====================
    def _hunyuan_download_rest_files_async(
        object_dir: Path,
        repo_root: Path,
        downloads: List[Dict[str, str]],
        object_dir_name: str,
        mode: str,
        result_format: str,
        batch_tag: str,
        session_id: str,
        registry,
    ):
        """后台异步下载 mesh 和其他非 preview 文件，ZIP 自动解压"""
        import zipfile

        _logger = logging.getLogger(__name__)
        try:
            mesh_count = 0
            for it in downloads:
                url = str(it.get("url", "")).strip()
                if not url:
                    continue

                ext = os.path.splitext(_filename_from_url(url))[1].lower() or ".bin"

                if ext in {".webp", ".png", ".jpg", ".jpeg"}:
                    continue

                # ---- 判断类型 ----
                is_zip = ext == ".zip"
                is_mesh = ext in {".glb", ".gltf", ".obj", ".fbx", ".stl", ".usdz"}

                if is_mesh:
                    typ = "mesh"
                    mesh_count += 1
                    preferred = f"base{ext}"
                    output_type = "file"
                elif is_zip:
                    typ = "archive"
                    preferred = f"model_{batch_tag}.zip"
                    output_type = "file"
                else:
                    typ = ext.lstrip(".")
                    preferred = f"{typ}_{batch_tag}{ext}"
                    output_type = "file"

                try:
                    local_path = _download_url_to_dir(
                        url,
                        str(object_dir),
                        timeout=float(hunyuan_config.request_timeout),
                        preferred_filename=preferred,
                    )

                    if is_zip:
                        # ---- ZIP 解压：解压到目录后删除 ZIP ----
                        _logger.info(f"混元3D解压ZIP: {local_path}")
                        extracted_files = []
                        try:
                            with zipfile.ZipFile(local_path, "r") as zf:
                                for member in zf.namelist():
                                    # 跳过目录条目和隐藏文件
                                    if member.endswith("/") or member.startswith("__MACOSX"):
                                        continue
                                    # 安全解压：只取文件名，防止路径穿越
                                    orig_name = os.path.basename(member)
                                    if not orig_name:
                                        continue
                                    member_ext = os.path.splitext(orig_name)[1].lower()
                                    # 重命名为语义化文件名
                                    if member_ext in {".obj", ".glb", ".gltf", ".fbx", ".stl", ".usdz"}:
                                        safe_name = f"base{member_ext}"
                                    elif member_ext == ".mtl":
                                        safe_name = "base.mtl"
                                    else:
                                        safe_name = _safe_filename(orig_name)
                                    if not safe_name:
                                        continue
                                    dest_path = os.path.join(str(object_dir), safe_name)
                                    # 避免覆盖已存在的同名文件（如 base.glb）
                                    if os.path.exists(dest_path):
                                        base_n, ext_n = os.path.splitext(safe_name)
                                        dest_path = os.path.join(str(object_dir), f"{base_n}_from_zip{ext_n}")
                                    with zf.open(member) as src, open(dest_path, "wb") as dst:
                                        dst.write(src.read())
                                    extracted_files.append(dest_path)

                            # 删除原始 ZIP
                            try:
                                os.remove(local_path)
                            except Exception:
                                pass

                            # 注册解压出的文件
                            for ef in extracted_files:
                                ef_ext = os.path.splitext(ef)[1].lower()
                                relative_path = _to_repo_relative_path(ef, repo_root)

                                if ef_ext in {".obj", ".glb", ".gltf", ".fbx", ".stl", ".usdz"}:
                                    ef_type = "file"
                                    ef_short = "obj_model"
                                    mesh_count += 1
                                elif ef_ext in {".mtl"}:
                                    ef_type = "file"
                                    ef_short = "material"
                                elif ef_ext in {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff"}:
                                    ef_type = "file"
                                    ef_short = "texture"
                                else:
                                    ef_type = "file"
                                    ef_short = ef_ext.lstrip(".") or "misc"

                                file_id = registry.register(
                                    session_id=session_id,
                                    content_url=str(Path(ef).resolve()),
                                    resource_type=ef_type,
                                    content_text=relative_path,
                                    parameter={
                                        "additional_type": ["hunyuan_3d"],
                                        "object_id": object_dir_name,
                                        "mode": mode,
                                        "result_format": result_format,
                                        "name": os.path.basename(ef),
                                        "short_id": ef_short,
                                    },
                                )
                                _logger.debug(f"混元3D ZIP解压注册: {ef} (file_id={file_id})")

                            _logger.info(f"混元3D ZIP解压完成: {len(extracted_files)} 个文件")

                        except zipfile.BadZipFile:
                            _logger.warning(f"混元3D ZIP文件损坏，保留原文件: {local_path}")
                            relative_path = _to_repo_relative_path(local_path, repo_root)
                            registry.register(
                                session_id=session_id,
                                content_url=str(Path(local_path).resolve()),
                                resource_type=output_type,
                                content_text=relative_path,
                                parameter={
                                    "additional_type": ["hunyuan_3d"],
                                    "object_id": object_dir_name,
                                    "mode": mode,
                                    "result_format": result_format,
                                    "name": it.get("name"),
                                    "short_id": "archive",
                                },
                            )
                    else:
                        # ---- 非 ZIP：正常注册 ----
                        relative_path = _to_repo_relative_path(local_path, repo_root)
                        file_id = registry.register(
                            session_id=session_id,
                            content_url=str(Path(local_path).resolve()),
                            resource_type=output_type,
                            content_text=relative_path,
                            parameter={
                                "additional_type": ["hunyuan_3d"],
                                "object_id": object_dir_name,
                                "mode": mode,
                                "result_format": result_format,
                                "name": it.get("name"),
                                "short_id": "base" if typ == "mesh" else typ,
                            },
                        )
                        _logger.debug(f"混元3D后台下载+注册完成: {url} -> {relative_path} (file_id={file_id})")

                except Exception as e:
                    _logger.warning(f"混元3D后台下载/注册失败: {url}, err={e}")

            _logger.info(f"混元3D后台异步下载完成: {object_dir}, mesh count={mesh_count}")

        except Exception as e:
            _logger.error(f"混元3D后台下载异常: {e}")
        finally:
            _signal_mesh_ready(object_dir_name)

    # ==================== 主工具函数 ====================
    def _hunyuan_generate_3d(
        mode: str = "image_to_3d",
        images: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        result_format: str = "",
        enable_pbr: bool = False,
        generate_type: str = "",
        model_version: str = "",
        face_count: Optional[int] = None,
        download_dir: Optional[str] = None,
    ) -> str:
        _logger = logging.getLogger(__name__)
        try:
            repo_root = _get_active_project_path().resolve()
            assets_model_root = get_project_models_dir().resolve()

            mode = (mode or "").strip()

            # 解析 fileid:// -> http(s) url
            image_list: List[str] = []
            for image in images or []:
                if isinstance(image, str) and image.startswith("fileid://"):
                    file_id = image[9:].strip()
                    image_list.append(media_registry.resolve(file_id))
                else:
                    image_list.append(image)

            prompt = prompt.strip() if isinstance(prompt, str) else None
            if mode not in {"image_to_3d", "text_to_3d"}:
                raise ValueError("mode 必须是 image_to_3d 或 text_to_3d")

            if mode == "image_to_3d" and not image_list:
                raise ValueError("image_to_3d 模式必须提供 images")
            if mode == "text_to_3d" and not prompt:
                raise ValueError("text_to_3d 模式必须提供 prompt")

            actual_format = result_format or default_result_format
            actual_pbr = enable_pbr or default_enable_pbr
            actual_generate_type = generate_type or default_generate_type
            actual_model = model_version or default_model
            actual_face_count = face_count if face_count is not None else default_face_count

            result = client.run_to_download_urls(
                images=image_list if mode == "image_to_3d" else None,
                prompt=prompt if mode == "text_to_3d" else None,
                result_format=actual_format,
                enable_pbr=actual_pbr,
                face_count=actual_face_count,
                generate_type=actual_generate_type,
                model=actual_model,
                poll_interval=poll_interval,
                poll_timeout=poll_timeout,
            )

            _logger.info(
                "混元3D done job_id=%s downloads=%s",
                result.get("task_uuid"),
                len(result.get("downloads") or []),
            )

            # 下载到本地
            downloads = result.get("downloads") or []
            if not downloads:
                raise RuntimeError("混元3D 未返回任何可下载文件")

            # 目录名优先使用 prompt 文本（截取前30字符），否则用时间戳
            if prompt:
                dir_label = _safe_dirname(prompt[:30])
            else:
                dir_label = f"hunyuan_{time.strftime('%Y%m%d_%H%M%S')}"
            object_dir_name = dir_label or "模型"

            original_dir_name = object_dir_name
            suffix_idx = 1
            while (assets_model_root / object_dir_name).exists():
                object_dir_name = f"{original_dir_name}_{suffix_idx}"
                suffix_idx += 1

            object_dir = assets_model_root / object_dir_name
            object_dir.mkdir(parents=True, exist_ok=True)
            batch_tag = str(int(time.time()))

            registry = get_media_registry()
            session_id = get_current_session()

            preview_items = []
            rest_items = []

            for it in downloads:
                url = str(it.get("url", "")).strip()
                if not url:
                    continue
                ext = os.path.splitext(_filename_from_url(url))[1].lower() or ".bin"
                file_type = it.get("type", "")

                if ext in {".webp", ".png", ".jpg", ".jpeg"} or file_type == "IMAGE":
                    preview_items.append(it)
                else:
                    rest_items.append(it)

            # 同步下载 preview
            preview_parts = []
            preview_count = 0

            for it in preview_items:
                url = str(it.get("url", "")).strip()
                if not url:
                    continue

                ext = os.path.splitext(_filename_from_url(url))[1].lower() or ".png"
                preview_count += 1
                preferred = f"{preview_count:04d}{ext}"
                output_type = "image"

                try:
                    local_path = _download_url_to_dir(
                        url,
                        str(object_dir),
                        timeout=float(hunyuan_config.request_timeout),
                        preferred_filename=preferred,
                    )
                    relative_path = _to_repo_relative_path(local_path, repo_root)

                    file_id = registry.register(
                        session_id=session_id,
                        content_url=str(Path(local_path).resolve()),
                        resource_type=output_type,
                        content_text=relative_path,
                        parameter={
                            "additional_type": ["hunyuan_3d"],
                            "object_id": object_dir_name,
                            "mode": mode,
                            "result_format": actual_format,
                            "name": it.get("name"),
                            "short_id": str(preview_count),
                        },
                    )

                    preview_parts.append(
                        build_part(
                            content_type=output_type,
                            content_text=relative_path,
                            file_id=file_id,
                        )
                    )

                    _logger.debug(f"混元3D预览图下载完成: {url} -> {relative_path}")

                except Exception as e:
                    _logger.warning(f"混元3D预览图下载失败: {url}, err={e}")

            if not preview_parts and not rest_items:
                raise RuntimeError("未能获取任何下载资源")

            model_folder_relative = _to_repo_relative_path(str(object_dir), repo_root)

            # 后台异步下载 mesh
            if rest_items:
                _register_mesh_event(object_dir_name)
                bg_thread = threading.Thread(
                    target=_hunyuan_download_rest_files_async,
                    args=(
                        object_dir,
                        repo_root,
                        rest_items,
                        object_dir_name,
                        mode,
                        actual_format,
                        batch_tag,
                        session_id,
                        registry,
                    ),
                    daemon=True,
                )
                bg_thread.start()
                _logger.info(
                    f"混元3D后台异步任务已启动: {object_dir}, 将下载并注册 {len(rest_items)} 个资源"
                )

            return build_success_result(
                parts=preview_parts,
                metadata={
                    "model_folder": model_folder_relative,
                    "object_id": object_dir_name,
                    "task_uuid": result.get("task_uuid"),
                    "preview_count": len(preview_parts),
                    "has_mesh_pending": len(rest_items) > 0,
                },
            ).to_envelope(interface_type="media")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="media"
            )

    return [
        StructuredTool(
            name="hunyuan_generate_3d",
            description="调用腾讯混元3D API 生成 3D（image_to_3d / text_to_3d）",
            func=_hunyuan_generate_3d,
            args_schema=Hunyuan3DGenerate3DInput,
        )
    ]
