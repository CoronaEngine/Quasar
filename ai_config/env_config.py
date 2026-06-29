from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value.strip()


def _parse_env_value(value: str) -> str:
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()

    name, separator, value = line.partition("=")
    if not separator:
        return None
    name = name.strip()
    if not _ENV_NAME_RE.match(name):
        return None
    return name, _parse_env_value(value)


def load_env_file(
    path: str | os.PathLike[str],
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into the process environment."""
    env = os.environ if environ is None else environ
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        name, value = parsed
        if name in loaded:
            continue
        if not override and name in env:
            continue
        env[name] = value
        loaded[name] = value
    return loaded


def _env(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _set_if_value(target: dict[str, Any], key: str, value: str) -> None:
    if value:
        target[key] = value


def _add_provider(
    providers: list[dict[str, Any]],
    *,
    name: str,
    provider_type: str,
    api_key: str,
    base_url: str,
) -> None:
    if not any((api_key, base_url)):
        return
    entry: dict[str, Any] = {
        "name": name,
        "type": provider_type,
    }
    if base_url:
        entry["base_url"] = base_url
    if api_key:
        entry["api_key"] = api_key
    providers.append(entry)


def build_env_ai_settings(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build Quasar AI settings from generic environment variables."""
    env = os.environ if environ is None else environ
    settings: dict[str, Any] = {}
    providers: list[dict[str, Any]] = []

    common_provider = _env(env, "AI_PROVIDER_NAME") or "doubao"
    common_type = _env(env, "AI_PROVIDER_TYPE") or "openai-compatible"
    common_key = _env(env, "AI_API_KEY")
    common_base_url = _env(env, "AI_BASE_URL")
    llm_model = _env(env, "LLM_MODEL")
    tool_model = _env(env, "TOOL_MODEL")

    _add_provider(
        providers,
        name=common_provider,
        provider_type=common_type,
        api_key=common_key,
        base_url=common_base_url,
    )
    if any((common_key, common_base_url, llm_model)):
        settings["chat"] = {
            "provider": common_provider,
            "model": llm_model or "gpt-4",
        }
    if any((common_key, common_base_url, llm_model, tool_model)):
        settings["tool_models"] = {
            "mcp": {
                "provider": common_provider,
                "model": tool_model or llm_model or "gpt-4",
            }
        }

    omni_provider = _env(env, "OMNI_PROVIDER_NAME") or "omni"
    omni_key = _env(env, "OMNI_API_KEY")
    omni_base_url = _env(env, "OMNI_BASE_URL")
    omni_model = _env(env, "OMNI_MODEL")
    _add_provider(
        providers,
        name=omni_provider,
        provider_type="openai-compatible",
        api_key=omni_key,
        base_url=omni_base_url,
    )
    if any((omni_key, omni_base_url, omni_model)):
        settings["omni"] = {
            "enable": True,
            "provider": omni_provider,
            "model": omni_model or llm_model or "omni-model",
        }

    image_provider = _env(env, "IMAGE_PROVIDER_NAME") or "dmx_image"
    image_key = _env(env, "IMAGE_API_KEY")
    image_base_url = _env(env, "IMAGE_BASE_URL")
    image_model = _env(env, "IMAGE_MODEL")
    image_effective_base_url = image_base_url
    if not image_effective_base_url and image_provider == "dmx_image" and image_key:
        image_effective_base_url = "https://www.dmxapi.cn/v1/images/generations"
    _add_provider(
        providers,
        name=image_provider,
        provider_type="openai-compatible",
        api_key=image_key,
        base_url=image_effective_base_url,
    )
    if any((image_key, image_base_url, image_model)):
        settings["image"] = {
            "enable": True,
            "provider": image_provider,
            "model": image_model or "gpt-image-2-ssvip",
        }
        if image_effective_base_url:
            settings["image"]["base_url"] = image_effective_base_url

    video_provider = _env(env, "VIDEO_PROVIDER_NAME") or "video"
    video_key = _env(env, "VIDEO_API_KEY")
    video_base_url = _env(env, "VIDEO_BASE_URL")
    video_model = _env(env, "VIDEO_MODEL")
    _add_provider(
        providers,
        name=video_provider,
        provider_type="openai-compatible",
        api_key=video_key,
        base_url=video_base_url,
    )
    if any((video_key, video_base_url, video_model)):
        settings["video"] = {
            "enable": True,
            "provider": video_provider,
            "model": video_model or "video-model",
        }
        if video_base_url:
            settings["video"]["base_url"] = video_base_url

    music_key = _env(env, "MUSIC_API_KEY")
    music_base_url = _env(env, "MUSIC_BASE_URL")
    if any((music_key, music_base_url)):
        settings["music"] = {
            "api_key": music_key,
            "base_url": music_base_url or "https://api.sunoapi.org",
        }

    hunyuan_key = _env(env, "HUNYUAN3D_API_KEY")
    hunyuan_keys = _split_csv(_env(env, "HUNYUAN3D_API_KEYS"))
    hunyuan_config: dict[str, Any] = {}
    if any((hunyuan_key, hunyuan_keys)):
        hunyuan_config["enable"] = True
        hunyuan_config["api_key"] = hunyuan_key
        hunyuan_config["api_keys"] = hunyuan_keys
    _set_if_value(hunyuan_config, "region", _env(env, "HUNYUAN3D_REGION"))
    _set_if_value(hunyuan_config, "endpoint", _env(env, "HUNYUAN3D_ENDPOINT"))
    _set_if_value(hunyuan_config, "version", _env(env, "HUNYUAN3D_VERSION"))
    _set_if_value(hunyuan_config, "result_format", _env(env, "HUNYUAN3D_RESULT_FORMAT"))
    _set_if_value(hunyuan_config, "enable_pbr", _env(env, "HUNYUAN3D_ENABLE_PBR"))
    _set_if_value(hunyuan_config, "model", _env(env, "HUNYUAN3D_MODEL"))
    _set_if_value(hunyuan_config, "generate_type", _env(env, "HUNYUAN3D_GENERATE_TYPE"))
    _set_if_value(hunyuan_config, "face_count", _env(env, "HUNYUAN3D_FACE_COUNT"))
    _set_if_value(
        hunyuan_config,
        "request_timeout",
        _env(env, "HUNYUAN3D_REQUEST_TIMEOUT"),
    )
    _set_if_value(hunyuan_config, "poll_interval", _env(env, "HUNYUAN3D_POLL_INTERVAL"))
    _set_if_value(hunyuan_config, "poll_timeout", _env(env, "HUNYUAN3D_POLL_TIMEOUT"))
    if hunyuan_config:
        settings["hunyuan3d"] = hunyuan_config

    dashscope_key = _env(env, "DASHSCOPE_API_KEY", "DASH_SCOPE_API_KEY")
    if dashscope_key:
        settings["object_recognition"] = {
            "enable": True,
            "provider": "dashscope",
            "dashscope_api_key": dashscope_key,
        }

    if providers:
        settings["providers"] = providers
    return settings


def apply_env_ai_settings(
    environ: Mapping[str, str] | None = None,
    *,
    collector: Any | None = None,
) -> dict[str, Any]:
    """Register environment-derived settings with Quasar's config collector."""
    settings = build_env_ai_settings(environ)
    if not settings:
        return {}

    if collector is None:
        from ..ai_service.entrance import ai_entrance

        collector = ai_entrance.collector

    ordered_keys = [
        "providers",
        "chat",
        "tool_models",
        "omni",
        "image",
        "video",
        "music",
        "hunyuan3d",
        "object_recognition",
    ]
    for key in [name for name in ordered_keys if name in settings]:
        value = settings[key]
        snapshot = copy.deepcopy(value)

        def setting(snapshot=snapshot):
            return copy.deepcopy(snapshot)

        setting.__module__ = __name__
        collector.register_setting(key)(setting)

    return copy.deepcopy(settings)


__all__ = [
    "apply_env_ai_settings",
    "build_env_ai_settings",
    "load_env_file",
]
