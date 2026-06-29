from __future__ import annotations

import os
import sys
from pathlib import Path

QUASAR_DIR = Path(__file__).resolve().parents[1]
AITOOL_DIR = QUASAR_DIR.parent
for candidate in (AITOOL_DIR,):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from Quasar.ai_config.env_config import (
    apply_env_ai_settings,
    build_env_ai_settings,
    load_env_file,
)
from Quasar.ai_modules.three_d_generate.tools.loader import _load_hunyuan_3d_config


def test_load_env_file_parses_comments_quotes_empty_values_and_keeps_first_duplicate(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "AI_API_KEY='key-from-file'",
                'AI_BASE_URL="https://example.test/v1"',
                "EMPTY_VALUE=",
                "AI_API_KEY=second-value",
                "INLINE_COMMENT=value # trailing comment",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    monkeypatch.delenv("EMPTY_VALUE", raising=False)
    monkeypatch.delenv("INLINE_COMMENT", raising=False)

    loaded = load_env_file(env_file)

    assert loaded == {
        "AI_API_KEY": "key-from-file",
        "AI_BASE_URL": "https://example.test/v1",
        "EMPTY_VALUE": "",
        "INLINE_COMMENT": "value",
    }
    assert os.environ["AI_API_KEY"] == "key-from-file"
    assert os.environ["AI_BASE_URL"] == "https://example.test/v1"
    assert os.environ["EMPTY_VALUE"] == ""
    assert os.environ["INLINE_COMMENT"] == "value"


def test_load_env_file_does_not_override_existing_process_environment(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AI_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setenv("AI_API_KEY", "process-value")

    loaded = load_env_file(env_file)

    assert loaded == {}
    assert os.environ["AI_API_KEY"] == "process-value"


def test_build_env_ai_settings_generates_common_provider_and_model_settings() -> None:
    settings = build_env_ai_settings(
        {
            "AI_API_KEY": "common-key",
            "AI_BASE_URL": "https://llm.example/v1",
            "AI_PROVIDER_NAME": "main",
            "AI_PROVIDER_TYPE": "openai-compatible",
            "LLM_MODEL": "gpt-env",
            "TOOL_MODEL": "gpt-tool-env",
        }
    )

    assert settings["providers"] == [
        {
            "name": "main",
            "type": "openai-compatible",
            "base_url": "https://llm.example/v1",
            "api_key": "common-key",
        }
    ]
    assert settings["chat"]["provider"] == "main"
    assert settings["chat"]["model"] == "gpt-env"
    assert settings["tool_models"]["mcp"] == {"provider": "main", "model": "gpt-tool-env"}


def test_build_env_ai_settings_overrides_media_and_music_settings() -> None:
    settings = build_env_ai_settings(
        {
            "AI_API_KEY": "common-key",
            "AI_BASE_URL": "https://llm.example/v1",
            "OMNI_API_KEY": "omni-key",
            "OMNI_BASE_URL": "https://omni.example/v1",
            "OMNI_MODEL": "omni-env",
            "IMAGE_API_KEY": "image-key",
            "IMAGE_BASE_URL": "https://image.example/v1/images/generations",
            "IMAGE_MODEL": "image-env",
            "MUSIC_API_KEY": "music-key",
            "MUSIC_BASE_URL": "https://music.example",
        }
    )

    provider_by_name = {entry["name"]: entry for entry in settings["providers"]}
    assert provider_by_name["omni"]["api_key"] == "omni-key"
    assert provider_by_name["dmx_image"]["api_key"] == "image-key"
    assert settings["omni"]["provider"] == "omni"
    assert settings["omni"]["model"] == "omni-env"
    assert settings["image"]["provider"] == "dmx_image"
    assert settings["image"]["model"] == "image-env"
    assert settings["music"]["api_key"] == "music-key"
    assert settings["music"]["base_url"] == "https://music.example"


def test_image_api_key_keeps_builtin_dmx_defaults_when_optional_values_are_missing() -> None:
    settings = build_env_ai_settings({"IMAGE_API_KEY": "image-key"})

    assert settings["providers"] == [
        {
            "name": "dmx_image",
            "type": "openai-compatible",
            "base_url": "https://www.dmxapi.cn/v1/images/generations",
            "api_key": "image-key",
        }
    ]
    assert settings["image"]["provider"] == "dmx_image"
    assert settings["image"]["model"] == "gpt-image-2-ssvip"
    assert settings["image"]["base_url"] == "https://www.dmxapi.cn/v1/images/generations"


def test_hunyuan3d_api_keys_from_env_are_split_for_loader() -> None:
    settings = build_env_ai_settings(
        {
            "HUNYUAN3D_API_KEYS": "key-a, key-b,, key-c",
            "HUNYUAN3D_MODEL": "3.1",
            "HUNYUAN3D_RESULT_FORMAT": "OBJ",
            "HUNYUAN3D_ENABLE_PBR": "true",
            "HUNYUAN3D_GENERATE_TYPE": "LowPoly",
            "HUNYUAN3D_FACE_COUNT": "300000",
            "HUNYUAN3D_REGION": "ap-shanghai",
            "HUNYUAN3D_ENDPOINT": "example.tencentmaas.com",
            "HUNYUAN3D_VERSION": "rapid",
            "HUNYUAN3D_REQUEST_TIMEOUT": "123.5",
            "HUNYUAN3D_POLL_INTERVAL": "4.5",
            "HUNYUAN3D_POLL_TIMEOUT": "456.5",
        }
    )

    hunyuan = _load_hunyuan_3d_config(settings["hunyuan3d"])

    assert hunyuan.api_keys == ["key-a", "key-b", "key-c"]
    assert hunyuan.api_key == ""
    assert hunyuan.model == "3.1"
    assert hunyuan.result_format == "OBJ"
    assert hunyuan.enable_pbr is True
    assert hunyuan.generate_type == "LowPoly"
    assert hunyuan.face_count == 300000
    assert hunyuan.region == "ap-shanghai"
    assert hunyuan.endpoint == "example.tencentmaas.com"
    assert hunyuan.version == "rapid"
    assert hunyuan.request_timeout == 123.5
    assert hunyuan.poll_interval == 4.5
    assert hunyuan.poll_timeout == 456.5


def test_build_env_ai_settings_uses_dashscope_key_for_object_recognition() -> None:
    settings = build_env_ai_settings({"DASH_SCOPE_API_KEY": "dash-key"})

    assert settings["object_recognition"]["provider"] == "dashscope"
    assert settings["object_recognition"]["dashscope_api_key"] == "dash-key"


def test_build_env_ai_settings_returns_empty_mapping_when_no_supported_env_exists() -> None:
    assert build_env_ai_settings({"UNRELATED": "value"}) == {}


def test_apply_env_ai_settings_registers_settings_on_supplied_collector() -> None:
    class CapturingCollector:
        def __init__(self) -> None:
            self.settings: dict[str, object] = {}

        def register_setting(self, key: str):
            def decorator(func):
                self.settings[key] = func()
                return func

            return decorator

    collector = CapturingCollector()

    applied = apply_env_ai_settings(
        {"AI_API_KEY": "key", "AI_BASE_URL": "https://example.test/v1"},
        collector=collector,
    )

    assert set(applied) == {"providers", "chat", "tool_models"}
    assert collector.settings["chat"]["provider"] == "doubao"
    assert collector.settings["providers"][0]["api_key"] == "key"
