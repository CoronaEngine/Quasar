from collections.abc import Callable, Iterator
from importlib import import_module
from typing import Any


class LazyRegistryRef:
    def __init__(self, module_name: str, getter_name: str):
        self._module_name = module_name
        self._getter_name = getter_name
        self._value = None

    def resolve(self):
        if self._value is None:
            module = import_module(self._module_name)
            self._value = getattr(module, self._getter_name)()
        return self._value

    def __getattr__(self, name: str):
        return getattr(self.resolve(), name)


def _load_default_ai_entrance():
    from ai_service import entrance

    entrance_cls = entrance.ai_entrance
    if not entrance_cls.if_import:
        entrance_cls.reimport()
    return entrance_cls


class CAIRuntime:
    def __init__(
        self,
        ai_entrance_provider: Callable[[], Any] | None = None,
        registries: dict[str, Any] | None = None,
    ):
        self._ai_entrance_provider = ai_entrance_provider or _load_default_ai_entrance
        self.registries = self._create_default_registries()
        if registries:
            self.registries.update(registries)
        self.plugins: list[Any] = []

    def get_ai_entrance(self):
        return self._ai_entrance_provider()

    def chat_stream(self, payload: dict) -> Iterator[str]:
        handler = getattr(self.get_ai_entrance(), "handle_integrated_entrance_stream")
        yield from handler(payload)

    def get_registry(self, name: str):
        registry = self.registries[name]
        if isinstance(registry, LazyRegistryRef):
            return registry.resolve()
        return registry

    def register_plugin(self, plugin: Any) -> None:
        register = getattr(plugin, "register", None)
        if register is None:
            raise TypeError("CAI plugin must expose register(runtime)")
        register(self)
        self.plugins.append(plugin)

    def shutdown(self) -> None:
        for plugin in reversed(self.plugins):
            shutdown = getattr(plugin, "shutdown", None)
            if shutdown is not None:
                shutdown(self)

    @staticmethod
    def _create_default_registries() -> dict[str, Any]:
        return {
            "config": LazyRegistryRef("ai_config.ai_config", "get_ai_config"),
            "tool": LazyRegistryRef("ai_tools.registry", "get_tool_registry"),
            "workflow": LazyRegistryRef("ai_workflow.registry", "get_workflow_registry"),
            "workflow_command": LazyRegistryRef(
                "ai_workflow.command_registry",
                "get_workflow_command_registry",
            ),
            "media": LazyRegistryRef("ai_media_resource", "get_media_registry"),
            "conversation": LazyRegistryRef(
                "ai_agent.conversation_store",
                "get_conversation_store",
            ),
            "model": LazyRegistryRef("ai_models.base_pool", "get_pool_registry"),
        }


_DEFAULT_RUNTIME: CAIRuntime | None = None


def get_default_runtime() -> CAIRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = CAIRuntime()
    return _DEFAULT_RUNTIME


def set_default_runtime(runtime: CAIRuntime | None) -> None:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = runtime