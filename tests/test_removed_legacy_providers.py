from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _module_names(relative: str) -> set[str]:
    tree = ast.parse(_source(relative))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_3d_tools_only_exports_hunyuan_loader() -> None:
    source = _source("ai_modules/three_d_generate/tools/__init__.py")

    assert "load_hunyuan3d_tools" in source
    assert "load_3d_tools" not in source


def test_model_tools_no_longer_registers_rodin_tool() -> None:
    source = _source("ai_modules/three_d_generate/tools/model_tools.py")
    names = _module_names("ai_modules/three_d_generate/tools/model_tools.py")

    assert "Hunyuan3DGenerate3DInput" in names
    assert "load_hunyuan3d_tools" in names
    assert "RodinGenerate3DInput" not in names
    assert "load_3d_tools" not in names
    assert "Rodin3DClient" not in source
    assert "rodin_generate_3d" not in source


def test_hunyuan_generation_is_limited_to_one_per_account() -> None:
    source = _source("ai_modules/three_d_generate/tools/model_tools.py")

    assert "per_key_concurrent = 1" in source
    assert "max_concurrent_generations', 3" not in source


def test_3d_config_no_longer_registers_rodin() -> None:
    dataclasses = _source("ai_modules/three_d_generate/configs/dataclasses.py")
    settings = _source("ai_modules/three_d_generate/configs/settings.py")
    loader = _source("ai_modules/three_d_generate/tools/loader.py")
    config_init = _source("ai_modules/three_d_generate/configs/__init__.py")

    assert "Hunyuan3DSettings" in dataclasses
    assert "Rodin3DSettings" not in dataclasses
    assert 'register_setting("rodin3d")' not in settings
    assert 'register_loader("rodin3d")' not in loader
    assert "Rodin3DSettings" not in config_init


def test_legacy_image_fallback_uses_dmx_only() -> None:
    source = _source("ai_models/base_pool/legacy_fallback.py")

    assert "DmxImageClient" in source
    assert "LingyaImageClient" not in source
    assert '"dmx_image"' in source
    assert "lingya_image" not in source
    assert ".get(key, " not in source


def test_image_warmup_uses_dmx_http_client() -> None:
    source = _source("ai_tools/warmup.py")

    assert "_get_dmx_http_client" in source
    assert "_get_image_http_client" not in source
    assert "client_image" not in source
