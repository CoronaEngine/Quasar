from __future__ import annotations

from Quasar.ai_modules.three_d_generate.tools.client_hunyuan3d import (
    sanitize_hunyuan_log_payload,
    sanitize_url_for_log,
)


def test_sanitize_url_for_log_removes_signed_query() -> None:
    url = (
        "https://hunyuan-prod.example.com/3d/output/model.glb"
        "?q-signature=secret&q-ak=akid&q-key-time=1%3B2"
    )

    sanitized = sanitize_url_for_log(url)

    assert sanitized == "https://hunyuan-prod.example.com/3d/output/model.glb?<redacted>"
    assert "q-signature" not in sanitized
    assert "secret" not in sanitized
    assert "q-ak" not in sanitized


def test_sanitize_hunyuan_log_payload_redacts_nested_urls() -> None:
    payload = {
        "ResultFile3Ds": [
            {
                "Url": "https://hunyuan.example.com/model.glb?q-signature=secret",
                "PreviewImageUrl": "https://hunyuan.example.com/preview.png?q-ak=akid",
            }
        ],
        "plain": "keep-me",
    }

    sanitized = sanitize_hunyuan_log_payload(payload)

    text = str(sanitized)
    assert "q-signature" not in text
    assert "q-ak" not in text
    assert "secret" not in text
    assert sanitized["plain"] == "keep-me"
