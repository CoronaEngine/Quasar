from __future__ import annotations

import logging
import os
import time
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ai_config.ai_config import AIConfig
from ai_media_resource import get_media_registry
from ai_tools.response_adapter import build_part, build_success_result, build_error_result
from config.app_config import get_app_config

from ai_modules.three_d_generate.tools.client_3d import Rodin3DClient


import re
import urllib.parse

_WIN_INVALID = r'[<>:"/\\|?*\x00-\x1F]'


def _safe_dirname(s: str) -> str:
    """Make a string safe to use as a directory name (Windows-friendly)."""
    s = (s or "").strip()
    # Normalize UUID like: 'xxxx - yyyy - ...' -> 'xxxx-yyyy-...'
    s = re.sub(r"\s*-\s*", "-", s)
    # Remove remaining whitespace
    s = re.sub(r"\s+", "", s)
    # Replace invalid characters
    s = re.sub(_WIN_INVALID, "_", s)
    return s or "task"


def _safe_filename(name: str) -> str:
    name = (name or "").strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r'[:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "download.bin"


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

    filename = _safe_filename(preferred_filename) if preferred_filename else _filename_from_url(url)
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

    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)
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
        description="可选：物体唯一标识，用于按 object_id 组织本地保存目录",
    )

    short_id: Optional[str] = Field(
        default=None,
        description="可选：用于输出文件命名的简短编号（例如 01/02）。不传则默认使用 01。",
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

    base_url = threed_config.base_url.strip()
    api_key = threed_config.api_key.strip()
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

    app_paths = get_app_config().paths
    repo_root = app_paths.repo_root.resolve()
    assets_model_root = app_paths.assets_model_dir.resolve()

    media_registry = get_media_registry()

    def _rodin_generate_3d(
        mode: str = "image_to_3d",
        images: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        object_id: Optional[str] = None,
        short_id: Optional[str] = None,
        condition_mode: str = "concat",
        tier: str = "Regular",
        quality: Optional[str] = None,
        seed: Optional[int] = None,
        geometry_file_format: str = "glb",
        material: Optional[str] = None,
        addons: Optional[str] = None,
        download_dir: Optional[str] = None,
    ) -> str:
        try:
            mode = (mode or "").strip()

            # 解析 fileid:// -> http(s) url
            image_list: List[str] = []
            for image in (images or []):
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

            logging.getLogger(__name__).info(
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
                logging.getLogger(__name__).warning(
                    "3D 保存目录已固定为 assets/model，忽略 download_dir 配置: arg=%s, cfg=%s, env=%s",
                    download_dir,
                    cfg_download_dir,
                    env_download_dir,
                )

            # ✅ 不用 task_uuid 建目录，改用 batch 目录（时间戳）
            object_dir_name = _safe_dirname(object_id or short_id or "object")
            object_dir = assets_model_root / object_dir_name
            object_dir.mkdir(parents=True, exist_ok=True)
            batch_tag = str(int(time.time()))


            # 输出文件命名：优先 short_id（例如 01/02）；不传则默认 01
            base_id = (short_id or "").strip() or "01"
            base_id = _safe_filename(base_id)

            downloads = result.get("downloads") or []
            if not downloads:
                raise RuntimeError("Rodin 未返回任何可下载文件（downloads 为空）")

            # 输出文件命名：如果调用方传了 short_id 就用它；否则后面会自动递增
            fixed_id = _safe_filename(str(short_id).strip()) if (short_id and str(short_id).strip()) else None
            idx = 1
            seen = set()   # 用于判断一组是否完整（mesh+preview）

            parts = []
            for it in downloads:
                url = str(it.get("url", "")).strip()
                if not url:
                    continue

                ext = os.path.splitext(_filename_from_url(url))[1].lower() or ".bin"

                # 归类：用于自动递增分组（mesh+preview 算一组）
                if ext in {".glb", ".gltf", ".obj", ".fbx"}:
                    typ = "mesh"
                elif ext in {".webp", ".png", ".jpg", ".jpeg"}:
                    typ = "preview"
                else:
                    typ = ext

                cur_id = fixed_id if fixed_id is not None else f"{idx:02d}"

                if typ == "mesh":
                    preferred = f"{object_dir_name}_{batch_tag}_{cur_id}{ext}"
                    output_type = "file"
                elif typ == "preview":
                    preferred = f"{object_dir_name}_{batch_tag}_{cur_id}_preview{ext}"
                    output_type = "image"
                else:
                    preferred = f"{object_dir_name}_{batch_tag}_{cur_id}_{typ}{ext}"
                    output_type = "file"

                local_path = _download_url_to_dir(
                    url,
                    str(object_dir),
                    timeout=float(threed_config.request_timeout),
                    preferred_filename=preferred,
                )
                relative_path = _to_repo_relative_path(local_path, repo_root)

                parts.append(
                    build_part(
                        content_type=output_type,
                        content_text=relative_path,
                        parameter={
                            "additional_type": ["rodin_3d"],
                            "object_id": object_dir_name,
                            "mode": mode,
                            "geometry_file_format": geometry_file_format,
                            "tier": tier,
                            "quality": quality,
                            "name": it.get("name"),
                            "short_id": cur_id,
                        },
                    )
                )

                # ✅ 自动递增：当一个编号已经拿到 mesh+preview，就进入下一号
                if fixed_id is None:
                    seen.add(typ)
                    if "mesh" in seen and "preview" in seen:
                        idx += 1
                        seen.clear()


            if not parts:
                raise RuntimeError("Rodin downloads 中没有有效 url，无法下载")

            return build_success_result(parts=parts).to_envelope(interface_type="media")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="media")

    return [
        StructuredTool(
            name="rodin_generate_3d",
            description="调用 Rodin API 生成 3D（image_to_3d / text_to_3d）",
            func=_rodin_generate_3d,
            args_schema=RodinGenerate3DInput,
        )
    ]
