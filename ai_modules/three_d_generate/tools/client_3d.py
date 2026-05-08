import os
import tempfile
import time
import base64
import re
import logging
import threading
from typing import Any, Dict, List, Optional
import httpx


logger = logging.getLogger(__name__)


class Rodin3DClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self._generation_lock = threading.Lock()

        logger.info(
            "Initialized Rodin3DClient with base_url=%s api_key=%s",
            self.base_url,
            self.api_key[:8] + "****" if self.api_key else "(空)",
        )

    def _headers(self) -> Dict[str, str]:
        h = dict(self.extra_headers)
        h["Authorization"] = f"Bearer {self.api_key.strip()}"
        # h["Authorization"] = self.api_key
        return h

    # ------------------------------------------------------------------
    # ✅ 关键 1：输入图片归一化（对齐 video：支持 fileid / dataURI / url / file / path）
    # ------------------------------------------------------------------
    def _coerce_image_to_file_tuple(self, image_ref: str):
        """
        把各种图片输入统一成 (filename, bytes, mime)
        支持：
        - fileid://xxx        （系统内部引用 → 下载为本地文件）
        - http(s)://          （下载）
        - file://             （本地）
        - 本地路径
        - data:image/...base64
        """
        image_ref = (image_ref or "").strip()

        if not image_ref:
            raise ValueError("图片输入为空")

        # -------------------------------------------------
        # 1️⃣ fileid:// → 解析 → 下载成本地临时文件（核心）
        # -------------------------------------------------
        if image_ref.startswith("fileid://"):
            try:
                from ....ai_tools.response_adapter import resolve_parts

                resolved = resolve_parts(
                    [{"content_type": "file", "content_text": image_ref}],
                    timeout=30.0,
                )
                if not resolved:
                    raise ValueError("resolve_parts 返回空结果")

                url = resolved[0].get("content_url") or resolved[0].get("content_text")
                if not isinstance(url, str) or not url.startswith("http"):
                    raise ValueError(f"fileid 未解析为 URL: {image_ref}")

                # 下载
                with httpx.Client(timeout=self.timeout) as c:
                    r = c.get(url)
                    r.raise_for_status()
                    content_type = r.headers.get("content-type", "").lower()

                # 推断后缀
                suffix = ".bin"
                if "png" in content_type:
                    suffix = ".png"
                elif "jpeg" in content_type or "jpg" in content_type:
                    suffix = ".jpg"
                elif "webp" in content_type:
                    suffix = ".webp"

                fd, local_path = tempfile.mkstemp(prefix="rodin_", suffix=suffix)
                os.close(fd)
                with open(local_path, "wb") as f:
                    f.write(r.content)

                image_ref = local_path  # 🔥 关键：从此只走“本地文件”分支

            except Exception as e:
                raise ValueError(
                    f"无法处理 fileid 图片输入: {image_ref}, err={e}"
                ) from e

        # -------------------------------------------------
        # 2️⃣ data:image/...;base64
        # -------------------------------------------------
        if image_ref.startswith("data:image/"):
            m = re.match(
                r"^data:(image/[\w\+\-\.]+);base64,(.+)$", image_ref, re.I | re.S
            )
            if not m:
                raise ValueError("不合法的 data URI 图片输入")
            mime = m.group(1).lower()
            data = base64.b64decode(m.group(2))

            ext = ".png"
            if "jpeg" in mime or "jpg" in mime:
                ext = ".jpg"
            elif "webp" in mime:
                ext = ".webp"

            return f"image{ext}", data, mime

        # -------------------------------------------------
        # 3️⃣ http(s):// URL
        # -------------------------------------------------
        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            with httpx.Client(timeout=self.timeout) as c:
                r = c.get(image_ref)
                r.raise_for_status()
                mime = r.headers.get("content-type", "application/octet-stream")
                filename = os.path.basename(image_ref.split("?")[0]) or "image"
                return filename, r.content, mime

        # -------------------------------------------------
        # 4️⃣ file:// 本地路径
        # -------------------------------------------------
        if image_ref.startswith("file://"):
            image_ref = image_ref[len("file://"):]

        # -------------------------------------------------
        # 5️⃣ 本地路径
        # -------------------------------------------------
        if os.path.exists(image_ref):
            filename = os.path.basename(image_ref)
            ext = os.path.splitext(filename)[1].lower()
            mime = (
                "image/jpeg"
                if ext in [".jpg", ".jpeg"]
                else (
                    "image/png"
                    if ext == ".png"
                    else "image/webp" if ext == ".webp" else "application/octet-stream"
                )
            )
            with open(image_ref, "rb") as f:
                return filename, f.read(), mime

        raise ValueError(
            f"无法读取图片输入: {image_ref}. "
            "Rodin 需要上传图片二进制（multipart），"
            "请传入可访问 URL 或本地路径。"
        )

        # ------------------------------------------------------------------
        # 你原来的 submit/status/download 保持不变（如果没有就照你现有文件）
        # ------------------------------------------------------------------

    def submit_generation(
        self,
        *,
        generate_path: str,
        images: Optional[List[str]],
        form_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}{generate_path}"

        # images = [r"F:\GitHub\CoronaEngine\build\examples\engine\RelWithDebInfo\assets\fox\02.jpg"]

        form_fields = {
            k: v
            for k, v in (form_fields or {}).items()
            if v is not None and str(v).strip() != ""
        }

        # ✅ Rodin multipart：先用单图字段名 "image"（比 "images" 更常见）
        files = None
        if images:
            fn, data, mime = self._coerce_image_to_file_tuple(str(images[0]))
            files = [("images", (fn, data, mime))]  # 关键：image 而不是 images

        # files = {}
        # if images:
        #     # Rodin 是 multipart：把所有图片做成 files
        #     # 这里 key 名称要与你 Rodin API 要求一致（你已有实现就保持一致）
        #     file_tuples = []
        #     for img in images:
        #         fn, data, mime = self._coerce_image_to_file_tuple(str(img))
        #         file_tuples.append(("images", (fn, data, mime)))
        #     files = file_tuples

        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                endpoint,
                headers=self._headers(),
                data=form_fields,
                # files=files if files else None,
                files=files,
            )
            r.raise_for_status()
            return r.json()

    def check_status(
        self, *, status_path: str, subscription_key: str
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}{status_path}"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                endpoint,
                headers={
                    **self._headers(),
                    "Content-Type": "application/json",
                    "accept": "application/json",
                },
                json={"subscription_key": subscription_key},
            )
            r.raise_for_status()
            return r.json()

    def download(self, *, download_path: str, task_uuid: str) -> List[Dict[str, str]]:
        endpoint = f"{self.base_url}{download_path}"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                endpoint,
                headers={
                    **self._headers(),
                    "Content-Type": "application/json",
                    "accept": "application/json",
                },
                json={"task_uuid": task_uuid},
            )
            r.raise_for_status()
            raw = r.json()

        # 兼容不同的 API 返回格式
        if isinstance(raw, dict):
            if "data" in raw:
                data_node = raw.get("data")
                if isinstance(data_node, dict):
                    download_list = data_node.get("list") or data_node.get("files") or []
                elif isinstance(data_node, list):
                    download_list = data_node
                else:
                    download_list = []
            else:
                download_list = raw.get("list") or raw.get("files") or []
        elif isinstance(raw, list):
            download_list = raw
        else:
            download_list = []

        if not isinstance(download_list, list):
            raise RuntimeError(
                f"Rodin download 返回内容格式不合法: {type(download_list).__name__}"
            )

        items: List[Dict[str, str]] = []
        for it in download_list:
            if isinstance(it, str):
                url = it
                name = "output"
            elif isinstance(it, dict):
                url = it.get("url") or it.get("content_url") or it.get("file_url")
                name = it.get("name") or it.get("filename") or "output"
            else:
                continue

            if not url:
                continue

            url = url.strip()
            if url.startswith("fileid://"):
                # 兼容 fileid 直接返回
                url = f"{url}"
            elif url.startswith("//"):
                url = f"https:{url}"
            elif not url.startswith("http"):
                url = f"{self.base_url}{url}"

            items.append({"name": name, "url": url})

        return items

    # ------------------------------------------------------------------
    # ✅ 关键 2：补齐 integrated 需要的 run_to_download_urls
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_task_specific_jobs(
        status_response: Dict[str, Any], task_uuid: str
    ) -> List[Dict[str, Any]]:
        """尽量从状态返回中筛出当前 task 的 jobs，避免并发请求互相影响。"""
        job_list = status_response.get("jobs") or []
        if not isinstance(job_list, list):
            return []

        matched_jobs: List[Dict[str, Any]] = []
        task_uuid = str(task_uuid or "").strip()
        if not task_uuid:
            return []

        candidate_keys = (
            "task_uuid",
            "uuid",
            "job_uuid",
            "id",
            "request_uuid",
            "parent_uuid",
        )
        for job in job_list:
            if not isinstance(job, dict):
                continue
            if any(
                str(job.get(key, "")).strip() == task_uuid for key in candidate_keys
            ):
                matched_jobs.append(job)

        return matched_jobs

    def run_to_download_urls(
        self,
        *,
        generate_path: str,
        status_path: str,
        download_path: str,
        images: Optional[List[str]],
        form_fields: Dict[str, Any],
        poll_interval: float = 1.0,
        poll_timeout: float = 180.0,
    ) -> Dict[str, Any]:
        """
        integrated / tool wrapper 期望的统一接口：
        - 提交任务
        - 轮询状态
        - 完成后返回 downloads 列表
        """
        with self._generation_lock:
            submit = self.submit_generation(
                generate_path=generate_path,
                images=images,
                form_fields=form_fields,
            )

            task_uuid = submit.get("uuid") or submit.get("task_uuid")
            jobs = submit.get("jobs") or {}
            subscription_key = (
                jobs.get("subscription_key") if isinstance(jobs, dict) else None
            )

            if not task_uuid or not subscription_key:
                raise RuntimeError(
                    f"Rodin 提交返回缺少 uuid/subscription_key: {submit}"
                )

            logger.info(
                "Rodin submit accepted task_uuid=%s subscription_key=%s",
                task_uuid,
                subscription_key,
            )

            start = time.time()
            last_statuses = None  # 记录上次状态，避免重复日志
            while True:
                if time.time() - start > poll_timeout:
                    raise TimeoutError(
                        f"Rodin 任务超时（>{poll_timeout}s），task_uuid={task_uuid}"
                    )

                st = self.check_status(
                    status_path=status_path, subscription_key=subscription_key
                )
                matched_jobs = self._extract_task_specific_jobs(st, str(task_uuid))
                job_list = matched_jobs or (st.get("jobs") or [])
                statuses = []
                for j in job_list:
                    if isinstance(j, dict):
                        statuses.append(j.get("status"))

                # ✅ 只在状态改变时才输出日志，避免大量重复日志
                if statuses != last_statuses:
                    last_statuses = statuses
                    # subscription_key 已天然隔离任务作用域，
                    # 匹配不到 task 级 job 时使用全部 subscription jobs 是正常路径。
                    logger.debug(
                        "Rodin status task_uuid=%s matched_jobs=%s total_jobs=%s statuses=%s",
                        task_uuid,
                        len(matched_jobs),
                        len(job_list),
                        statuses,
                    )

                if any(s == "Failed" for s in statuses):
                    raise RuntimeError(
                        f"Rodin 任务失败：task_uuid={task_uuid}, status={st}"
                    )

                if statuses and all(s == "Done" for s in statuses):
                    downloads = self.download(
                        download_path=download_path, task_uuid=task_uuid
                    )

                    # 轮询完成后下载接口可能有短暂一致性延迟
                    if not downloads:
                        logger.warning(
                            "Rodin download 返回空列表，重试 3 次，task_uuid=%s",
                            task_uuid,
                        )
                        for attempt in range(3):
                            time.sleep(2)
                            downloads = self.download(
                                download_path=download_path, task_uuid=task_uuid
                            )
                            if downloads:
                                break

                    if not downloads:
                        raise RuntimeError(
                            f"Rodin 下载结果为空：task_uuid={task_uuid}, status={st}"
                        )

                    return {
                        "task_uuid": task_uuid,
                        "subscription_key": subscription_key,
                        "downloads": downloads,
                    }

                time.sleep(poll_interval)
