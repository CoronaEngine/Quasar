import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    event_type: str
    payload: dict[str, Any]
    request_id: str | None = None
    session_id: str | None = None
    sequence: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None

    @classmethod
    def from_legacy_chunk(cls, chunk: str | dict[str, Any]) -> "StreamEvent":
        raw = chunk if isinstance(chunk, str) else None
        payload = json.loads(chunk) if isinstance(chunk, str) else dict(chunk)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        request_id = metadata.get("request_id")
        session_id = payload.get("session_id")
        return cls(
            event_type=cls._detect_event_type(payload, metadata),
            payload=payload,
            request_id=request_id if isinstance(request_id, str) else None,
            session_id=session_id if isinstance(session_id, str) else None,
            metadata=metadata,
            raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    def to_legacy_chunk(self) -> str:
        return self.raw or json.dumps(self.payload, ensure_ascii=False)

    @staticmethod
    def _detect_event_type(payload: dict[str, Any], metadata: dict[str, Any]) -> str:
        event_type = payload.get("event_type")
        if isinstance(event_type, str) and event_type:
            return event_type
        if metadata.get("stream_done"):
            return "done"
        if metadata.get("heartbeat"):
            return "heartbeat"
        if payload.get("error_code"):
            return "error"
        chunk_type = payload.get("chunk_type")
        return chunk_type if isinstance(chunk_type, str) and chunk_type else "data"