from dataclasses import dataclass, field
from typing import Any


@dataclass
class AIError:
    code: str
    message: str
    recoverable: bool = False
    detail: Any = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "detail": self.detail,
            "source": self.source,
            "metadata": self.metadata,
        }