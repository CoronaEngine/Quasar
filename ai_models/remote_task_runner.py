from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteTaskResult:
    task_id: str
    status: str
    payload: Any
    poll_count: int
    elapsed_seconds: float


class RemoteTaskRunner:
    """Shared submit -> poll -> result runner for remote provider jobs."""

    DEFAULT_SUCCESS_STATUSES = frozenset({"done", "completed", "succeed", "success", "succeeded"})
    DEFAULT_FAILURE_STATUSES = frozenset({"fail", "failed", "error", "cancelled", "canceled"})

    def __init__(
        self,
        *,
        provider_name: str,
        poll_interval: float,
        poll_timeout: float,
        sleep_func: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider_name = str(provider_name or "remote")
        self.poll_interval = max(float(poll_interval), 0.0)
        self.poll_timeout = max(float(poll_timeout), 0.0)
        self._sleep = sleep_func
        self._clock = clock

    def run(
        self,
        *,
        submit: Callable[[], str],
        poll: Callable[[str], Any],
        result: Callable[[Any], Any] | None = None,
        status: Callable[[Any], str] | None = None,
        error: Callable[[Any], str] | None = None,
        success_statuses: Iterable[str] | None = None,
        failure_statuses: Iterable[str] | None = None,
        provider_task_label: str = "remote task",
    ) -> RemoteTaskResult:
        task_id = str(submit())
        if not task_id:
            raise RuntimeError(f"{provider_task_label} submit did not return a task id")

        logger.info("[%s] submitted %s id=%s", self.provider_name, provider_task_label, task_id)
        start = self._clock()
        last_status = ""
        poll_count = 0
        success = {s.lower() for s in (success_statuses or self.DEFAULT_SUCCESS_STATUSES)}
        failure = {s.lower() for s in (failure_statuses or self.DEFAULT_FAILURE_STATUSES)}

        while True:
            elapsed = self._clock() - start
            if self.poll_timeout and elapsed > self.poll_timeout:
                raise TimeoutError(f"{provider_task_label} timed out after {self.poll_timeout}s, id={task_id}")

            payload = poll(task_id)
            poll_count += 1
            current_status = self._normalize_status(status(payload) if status else self._default_status(payload))

            if current_status != last_status:
                last_status = current_status
                logger.debug(
                    "[%s] %s status id=%s status=%s",
                    self.provider_name,
                    provider_task_label,
                    task_id,
                    current_status,
                )

            if current_status in failure:
                error_message = error(payload) if error else self._default_error(payload)
                raise RuntimeError(f"{provider_task_label} failed: id={task_id}, error={error_message or 'unknown'}")

            if current_status in success:
                final_payload = result(payload) if result else payload
                logger.info(
                    "[%s] %s completed id=%s polls=%d elapsed=%.2fs",
                    self.provider_name,
                    provider_task_label,
                    task_id,
                    poll_count,
                    self._clock() - start,
                )
                return RemoteTaskResult(
                    task_id=task_id,
                    status=current_status,
                    payload=final_payload,
                    poll_count=poll_count,
                    elapsed_seconds=self._clock() - start,
                )

            self._sleep(self.poll_interval)

    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        return str(value or "").strip().lower()

    @classmethod
    def _default_status(cls, payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        response = payload.get("Response") if isinstance(payload.get("Response"), dict) else {}
        return str(
            payload.get("status")
            or payload.get("Status")
            or response.get("Status")
            or ""
        )

    @classmethod
    def _default_error(cls, payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        response = payload.get("Response") if isinstance(payload.get("Response"), dict) else {}
        return str(
            payload.get("error_message")
            or payload.get("ErrorMessage")
            or response.get("ErrorMessage")
            or payload.get("error")
            or payload.get("Error")
            or ""
        )


__all__ = ["RemoteTaskRunner", "RemoteTaskResult"]
