import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml


logger = logging.getLogger(__name__)


class CAIPlugin(Protocol):
    name: str
    enabled: bool

    def register(self, runtime) -> dict[str, Any]:
        ...


@dataclass
class ModulePluginSpec:
    name: str
    enabled: bool = True
    description: str = ""
    module_base: str = "ai_modules"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModulePluginSpec":
        return cls(
            name=str(data.get("name", "")),
            enabled=bool(data.get("enabled", False)),
            description=str(data.get("description", "")),
            module_base=str(data.get("module_base", "ai_modules")),
        )


class LegacyModulePlugin:
    def __init__(self, spec: ModulePluginSpec, modules_path: Path):
        self.spec = spec
        self.name = spec.name
        self.enabled = spec.enabled
        self.description = spec.description
        self._modules_path = modules_path

    def register(self, runtime) -> dict[str, Any]:
        if not self.enabled:
            logger.debug("跳过禁用模块: %s", self.name)
            return {"name": self.name, "enabled": False, "loaded": [], "failed": []}

        module_dir = self._modules_path / self.name
        loaded: list[str] = []
        failed: list[str] = []

        self._try_import("configs", module_dir / "configs" / "settings.py", f"{self.spec.module_base}.{self.name}.configs.settings", loaded, failed)
        self._try_import("base", module_dir / "base.py", f"{self.spec.module_base}.{self.name}.base", loaded, failed)
        self._try_import("loader", module_dir / "tools" / "loader.py", f"{self.spec.module_base}.{self.name}.tools.loader", loaded, failed)

        result = {
            "name": self.name,
            "enabled": True,
            "loaded": loaded,
            "failed": failed,
        }
        runtime.metadata.setdefault("plugins", {})[self.name] = result
        return result

    def _try_import(self, kind: str, path: Path, module_path: str, loaded: list[str], failed: list[str]) -> None:
        if not path.exists():
            return
        try:
            importlib.import_module(module_path)
            loaded.append(kind)
        except Exception as exc:
            failed.append(f"{kind}:{self.name}({exc})")
            logger.error("✗ 导入%s模块失败 %s: %s", kind, self.name, exc)


class PluginManager:
    def __init__(self, runtime):
        self.runtime = runtime
        self.plugins: list[Any] = []

    def register(self, plugin) -> dict[str, Any]:
        register = getattr(plugin, "register", None)
        if register is None:
            raise TypeError("CAI plugin must expose register(runtime)")
        result = register(self.runtime)
        self.plugins.append(plugin)
        return result if isinstance(result, dict) else {"name": getattr(plugin, "name", "unknown")}

    def load_module_settings(self, config_path: str | Path, modules_path: str | Path) -> dict[str, Any]:
        config_path = Path(config_path)
        modules_path = Path(modules_path)
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}

        summary = {"configs": [], "base": [], "loader": [], "failed": [], "disabled": []}
        for module_data in config.get("modules", []):
            spec = ModulePluginSpec.from_dict(module_data)
            plugin = LegacyModulePlugin(spec, modules_path)
            result = self.register(plugin)
            if not spec.enabled:
                summary["disabled"].append(spec.name)
                continue
            for kind in result.get("loaded", []):
                summary[kind].append(spec.name)
            summary["failed"].extend(result.get("failed", []))

        self.runtime.metadata["module_import_summary"] = summary
        return summary

    def shutdown(self) -> None:
        for plugin in reversed(self.plugins):
            shutdown = getattr(plugin, "shutdown", None)
            if shutdown is not None:
                shutdown(self.runtime)