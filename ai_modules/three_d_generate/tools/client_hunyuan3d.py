"""
腾讯混元生3D API 客户端

通过 TokenHub 调用混元生3D接口，使用 Bearer Token 鉴权。

接口：
  - 提交: POST /v1/api/3d/submit
  - 查询: POST /v1/api/3d/query
"""

import base64
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class Hunyuan3DClient:
    """腾讯混元生3D API 客户端（TokenHub）"""

    def __init__(
        self,
        *,
        api_key: str,
        region: str = "ap-guangzhou",
        endpoint: str = "tokenhub.tencentmaas.com",
        timeout: float = 120.0,
        version: str = "pro",
    ):
        self.api_key = api_key
        self.region = region
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.version = version
        self._generation_lock = threading.Lock()

        logger.info(
            "Initialized Hunyuan3DClient endpoint=%s version=%s api_key=%s",
            self.endpoint,
            self.version,
            self.api_key[:8] + "****" if self.api_key else "(空)",
        )

    # ------------------------------------------------------------------
    # HTTP 请求
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://{self.endpoint}{path}"
        payload = json.dumps(body, ensure_ascii=False)
        logger.debug("混元3D 请求: %s body=%s", url, payload[:500])

        with httpx.Client(timeout=self.timeout, verify=True, trust_env=False) as client:
            r = client.post(url, headers=self._headers(), content=payload.encode("utf-8"))
            if r.status_code >= 400:
                logger.error(
                    "混元3D HTTP %d: url=%s body_size=%d response=%s",
                    r.status_code, url, len(payload), r.text[:2000],
                )
            r.raise_for_status()
            resp = r.json()

        logger.debug("混元3D 响应: %s", json.dumps(resp, ensure_ascii=False)[:1000])

        # 检查错误（兼容多种返回格式）
        if isinstance(resp, dict):
            # TokenHub 格式
            error = resp.get("error")
            if error:
                if isinstance(error, dict):
                    raise RuntimeError(f"混元3D API 错误: {error.get('message', error)}")
                raise RuntimeError(f"混元3D API 错误: {error}")

            # 腾讯云原生格式
            response = resp.get("Response", {})
            if response.get("Error"):
                err = response["Error"]
                raise RuntimeError(f"混元3D API 错误 [{err.get('Code')}]: {err.get('Message')}")

        return resp

    # ------------------------------------------------------------------
    # 图片输入归一化
    # ------------------------------------------------------------------
    def _coerce_image_to_base64(self, image_ref: str) -> str:
        image_ref = (image_ref or "").strip()
        if not image_ref:
            raise ValueError("图片输入为空")

        if image_ref.startswith("data:image/"):
            m = re.match(r"^data:image/[\w\+\-\.]+;base64,(.+)$", image_ref, re.I | re.S)
            if not m:
                raise ValueError("不合法的 data URI 图片输入")
            return m.group(1)

        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            with httpx.Client(timeout=self.timeout, trust_env=False) as c:
                r = c.get(image_ref)
                r.raise_for_status()
                raw_bytes = r.content
                return self._compress_if_needed(raw_bytes)

        if image_ref.startswith("file://"):
            image_ref = image_ref[len("file://"):]

        if os.path.exists(image_ref):
            with open(image_ref, "rb") as f:
                raw_bytes = f.read()
            return self._compress_if_needed(raw_bytes)

        raise ValueError(f"无法读取图片输入: {image_ref}")

    @staticmethod
    def _compress_if_needed(raw_bytes: bytes, max_bytes: int = 3 * 1024 * 1024) -> str:
        """如果图片原始大小超过 max_bytes 则压缩，确保 base64 后≤6MB，单边128~5000"""
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(raw_bytes))
            w, h = img.size

            # 检查最小分辨率
            if min(w, h) < 128:
                raise ValueError(f"图片分辨率过小: {w}x{h}，单边不小于128")

            need_resize = max(w, h) > 5000
            need_compress = len(raw_bytes) > max_bytes

            if not need_resize and not need_compress:
                return base64.b64encode(raw_bytes).decode("ascii")

            if img.mode == "RGBA":
                img = img.convert("RGB")

            # 缩小到单边不超过 5000
            if need_resize:
                ratio = 5000 / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

            # 尝试不同质量压缩，确保 base64 后 ≤ 6MB（原始 ≤ ~4.5MB）
            for quality in (90, 80, 70, 60):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                compressed = buf.getvalue()
                if len(compressed) <= max_bytes:
                    break

            # 如果还是太大，继续缩小分辨率
            while len(compressed) > max_bytes:
                cur_w, cur_h = img.size
                img = img.resize((int(cur_w * 0.7), int(cur_h * 0.7)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                compressed = buf.getvalue()

            logger.info(
                "图片处理: %d bytes -> %d bytes, %dx%d -> %dx%d",
                len(raw_bytes), len(compressed), w, h, img.size[0], img.size[1],
            )
            return base64.b64encode(compressed).decode("ascii")

        except ImportError:
            logger.warning("Pillow 未安装，无法处理图片，直接发送原始数据")
            return base64.b64encode(raw_bytes).decode("ascii")

    # ------------------------------------------------------------------
    # 模型名映射
    # ------------------------------------------------------------------
    def _get_model_name(self, model: str = "") -> str:
        """根据版本和 model 参数返回 TokenHub 的 model 名称"""
        # 专业版
        if self.version == "pro":
            if model == "3.1":
                return "hy-3d-3.1"
            return "hy-3d-3.0"
        # 极速版
        return "hy-3d-rapid"

    # ------------------------------------------------------------------
    # 提交任务
    # ------------------------------------------------------------------
    def submit_job(
        self,
        *,
        images: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        result_format: str = "GLB",
        enable_pbr: bool = False,
        face_count: Optional[int] = None,
        generate_type: str = "Normal",
        model: str = "3.0",
    ) -> str:
        body: Dict[str, Any] = {
            "model": self._get_model_name(model),
        }

        # 图生3D：传 ImageBase64 / ImageUrl / Prompt（PascalCase）
        if images and len(images) > 0:
            img_ref = images[0]
            if img_ref.startswith("http://") or img_ref.startswith("https://"):
                body["ImageUrl"] = img_ref
            else:
                body["ImageBase64"] = self._coerce_image_to_base64(img_ref)
        elif prompt:
            body["Prompt"] = prompt
        else:
            raise ValueError("必须提供 images 或 prompt")

        if result_format:
            body["ResultFormat"] = result_format
        if enable_pbr:
            body["EnablePBR"] = True
        if face_count is not None:
            body["FaceCount"] = face_count
        if generate_type and generate_type != "Normal":
            body["GenerateType"] = generate_type

        resp = self._post("/v1/api/3d/submit", body)

        # 提取 job id（兼容多种字段名）
        job_id = (
            resp.get("id")
            or resp.get("job_id")
            or resp.get("JobId")
            or (resp.get("Response", {}) or {}).get("JobId")
            or (resp.get("data", {}) or {}).get("id")
        )
        if not job_id:
            raise RuntimeError(f"混元3D 提交任务未返回任务ID: {resp}")

        logger.info("混元3D 任务已提交: model=%s job_id=%s", body["model"], job_id)
        return str(job_id)

    # ------------------------------------------------------------------
    # 查询任务
    # ------------------------------------------------------------------
    def query_job(self, job_id: str, model: str = "3.0") -> Dict[str, Any]:
        body = {
            "model": self._get_model_name(model),
            "id": job_id,
        }
        return self._post("/v1/api/3d/query", body)

    # ------------------------------------------------------------------
    # 提交 + 轮询 + 返回下载列表（统一接口）
    # ------------------------------------------------------------------
    def run_to_download_urls(
        self,
        *,
        images: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        result_format: str = "GLB",
        enable_pbr: bool = False,
        face_count: Optional[int] = None,
        generate_type: str = "Normal",
        model: str = "3.0",
        poll_interval: float = 3.0,
        poll_timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        统一接口：提交 -> 轮询 -> 返回下载列表。
        返回格式与 Rodin3DClient 保持一致：
        {"task_uuid": ..., "downloads": [{"name": ..., "url": ...}, ...]}
        """
        with self._generation_lock:
            job_id = self.submit_job(
                images=images, prompt=prompt,
                result_format=result_format, enable_pbr=enable_pbr,
                face_count=face_count, generate_type=generate_type,
                model=model,
            )

            start = time.time()
            last_status = None

            while True:
                if time.time() - start > poll_timeout:
                    raise TimeoutError(f"混元3D 任务超时（>{poll_timeout}s），id={job_id}")

                resp = self.query_job(job_id, model=model)

                # 兼容多种状态字段和大小写
                status_raw = (
                    resp.get("status")
                    or resp.get("Status")
                    or (resp.get("Response", {}) or {}).get("Status")
                    or ""
                )
                status = status_raw.lower()

                if status != last_status:
                    last_status = status
                    logger.debug("混元3D status id=%s status=%s", job_id, status)

                if status in ("fail", "failed", "error"):
                    error_msg = (
                        resp.get("error_message")
                        or resp.get("ErrorMessage")
                        or (resp.get("Response", {}) or {}).get("ErrorMessage")
                        or "未知错误"
                    )
                    raise RuntimeError(f"混元3D 任务失败: id={job_id}, error={error_msg}")

                if status in ("done", "completed", "succeed", "success"):
                    logger.info("混元3D 任务完成原始响应: id=%s resp=%s", job_id, json.dumps(resp, ensure_ascii=False)[:4000])
                    downloads = self._extract_downloads(resp)
                    if not downloads:
                        raise RuntimeError(f"混元3D 任务完成但无下载文件: id={job_id}, resp_keys={list(resp.keys())}")

                    logger.info("混元3D 任务完成: id=%s, downloads=%d", job_id, len(downloads))
                    return {"task_uuid": job_id, "downloads": downloads}

                time.sleep(poll_interval)

    def _extract_downloads(self, resp: Dict[str, Any]) -> List[Dict[str, str]]:
        """从查询响应中提取下载文件列表"""
        downloads: List[Dict[str, str]] = []

        raw_data = resp.get("data")
        data_dict = raw_data if isinstance(raw_data, dict) else {}
        data_list = raw_data if isinstance(raw_data, list) else None
        response = resp.get("Response") if isinstance(resp.get("Response"), dict) else {}

        # data 本身就是文件列表的情况
        result_files = data_list or (
            resp.get("result_files")
            or resp.get("ResultFile3Ds")
            or response.get("ResultFile3Ds")
            or resp.get("outputs")
            or resp.get("output")
            or data_dict.get("result_files")
            or data_dict.get("outputs")
            or data_dict.get("output")
            or data_dict.get("ResultFile3Ds")
            or []
        )

        if isinstance(result_files, list):
            for f3d in result_files:
                if not isinstance(f3d, dict):
                    # 如果是纯字符串 URL
                    if isinstance(f3d, str) and f3d.startswith("http"):
                        downloads.append({"name": "model", "url": f3d, "type": "GLB"})
                    continue
                # 遍历所有值找 URL
                url = f3d.get("Url") or f3d.get("url") or f3d.get("download_url") or ""
                file_type = f3d.get("Type") or f3d.get("type") or f3d.get("format") or ""
                preview_url = f3d.get("PreviewImageUrl") or f3d.get("preview_url") or ""

                if url:
                    downloads.append({"name": file_type.lower() or "model", "url": url, "type": file_type})
                elif not url:
                    # 尝试从任意值中找 http URL
                    for v in f3d.values():
                        if isinstance(v, str) and v.startswith("http"):
                            downloads.append({"name": "model", "url": v, "type": "GLB"})
                            break
                if preview_url:
                    downloads.append({"name": f"preview_{file_type.lower()}", "url": preview_url, "type": "IMAGE"})

        # 单文件返回格式（顶层或 data 内）
        if not downloads:
            single_url = (
                resp.get("output_url") or resp.get("result_url")
                or resp.get("download_url") or resp.get("model_url")
                or data_dict.get("output_url") or data_dict.get("result_url")
                or data_dict.get("download_url") or data_dict.get("model_url")
                or ""
            )
            if single_url:
                downloads.append({"name": "model", "url": single_url, "type": "GLB"})

        # data 是 dict 但还没提取到，遍历所有值找 URL
        if not downloads and isinstance(raw_data, dict):
            for k, v in raw_data.items():
                if isinstance(v, str) and v.startswith("http"):
                    downloads.append({"name": k, "url": v, "type": "GLB"})

        if not downloads:
            logger.error("混元3D 无法提取下载文件, data type=%s, data=%s",
                         type(raw_data).__name__,
                         json.dumps(raw_data, ensure_ascii=False)[:3000] if raw_data else "None")

        return downloads
