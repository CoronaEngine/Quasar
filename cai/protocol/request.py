from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatRequest:
    payload: dict[str, Any]
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, request: "ChatRequest | dict[str, Any]") -> "ChatRequest":
        if isinstance(request, cls):
            return request
        if not isinstance(request, dict):
            raise TypeError("ChatRequest requires a dict or ChatRequest")
        return cls.from_legacy(request)

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> "ChatRequest":
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        session_id = payload.get("session_id")
        return cls(
            payload=dict(payload),
            session_id=session_id if isinstance(session_id, str) else None,
            metadata=dict(metadata),
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ChatRequest":
        payload = {
            "session_id": session_id,
            "llm_content": [
                {
                    "role": "user",
                    "interface_type": "integrated",
                    "part": [
                        {
                            "content_type": "text",
                            "content_text": text,
                        }
                    ],
                }
            ],
            "metadata": metadata or {},
        }
        return cls.from_legacy(payload)

    def to_legacy_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(self.metadata)
        payload["metadata"] = metadata
        return payload