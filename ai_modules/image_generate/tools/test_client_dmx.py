from __future__ import annotations

from .client_dmx import DmxImageClient


def test_parse_response_accepts_b64_json_as_data_uri():
    value, mime_type = DmxImageClient._parse_response(
        {"data": [{"b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}]}
    )

    assert mime_type == "image/png"
    assert value.startswith("data:image/png;base64,")
    assert "iVBORw0KGgo" in value


def test_parse_response_accepts_url_payload():
    value, mime_type = DmxImageClient._parse_response(
        {"data": [{"url": "https://example.com/generated.png"}]}
    )

    assert mime_type == "image/png"
    assert value == "https://example.com/generated.png"
