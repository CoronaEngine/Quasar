from __future__ import annotations

import json

from typing import Dict, Generator, Optional

from ai_tools.common import build_success_response


def build_integrated_text_response(
    session_id: str,
    metadata: Dict[str, object],
    text: str,
) -> str:
    return build_success_response(
        interface_type="integrated",
        session_id=session_id,
        metadata=metadata,
        llm_content=[
            {
                "role": "assistant",
                "interface_type": "integrated",
                "part": [
                    {
                        "content_type": "text",
                        "content_text": text,
                        "content_url": "",
                        "parameter": {},
                    }
                ],
            }
        ],
    )


def single_stream_response(
    session_id: str,
    metadata: Dict[str, object],
    text: str,
) -> Generator[str, None, None]:
    yield build_integrated_text_response(session_id, metadata, text)


def inject_function_id_to_review_stream(
    stream: Generator[str, None, None],
    function_id: Optional[int],
) -> Generator[str, None, None]:
    if function_id is None:
        for chunk in stream:
            yield chunk
        return

    for chunk in stream:
        try:
            data = json.loads(chunk)
            llm_content = data.get("llm_content", [])
            if isinstance(llm_content, list):
                for entry in llm_content:
                    parts = entry.get("part", [])
                    if isinstance(parts, list):
                        for part in parts:
                            if not isinstance(part, dict):
                                continue
                            review = (part.get("parameter") or {}).get("review")
                            if isinstance(review, dict) and review.get("stage") == "pending":
                                if not review.get("function_id"):
                                    review["function_id"] = function_id
            yield json.dumps(data)
        except Exception:
            yield chunk


__all__ = [
    "build_integrated_text_response",
    "single_stream_response",
    "inject_function_id_to_review_stream",
]
