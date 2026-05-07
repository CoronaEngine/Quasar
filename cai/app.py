from collections.abc import Iterable, Iterator
from typing import Any

from .protocol import ChatRequest
from .runtime import CAIRuntime, get_default_runtime


class CAIApp:
    def __init__(self, runtime: CAIRuntime | None = None):
        self.runtime = runtime or get_default_runtime()

    @classmethod
    def from_legacy_entrance(cls, get_ai_entrance):
        return cls(CAIRuntime(ai_entrance_provider=get_ai_entrance))

    def chat_stream(self, request: ChatRequest | dict[str, Any]) -> Iterator[str]:
        chat_request = ChatRequest.from_any(request)
        yield from self.runtime.chat_stream(chat_request.to_legacy_payload())

    def chat(self, request: ChatRequest | dict[str, Any]) -> list[str]:
        return list(self.chat_stream(request))

    def register_tool(self, tool: Any) -> None:
        registry = self.runtime.get_registry("tool")
        register = getattr(registry, "register_tool", None) or getattr(registry, "register", None)
        if register is None:
            raise AttributeError("tool registry does not expose register/register_tool")
        register(tool)

    def register_tools(self, tools: Iterable[Any]) -> None:
        for tool in tools:
            self.register_tool(tool)

    def register_workflow(self, workflow: Any) -> None:
        registry = self.runtime.get_registry("workflow")
        register = getattr(registry, "register", None) or getattr(registry, "register_workflow", None)
        if register is None:
            raise AttributeError("workflow registry does not expose register/register_workflow")
        register(workflow)

    def register_plugin(self, plugin: Any) -> None:
        self.runtime.register_plugin(plugin)

    def reset_session(self, session_id: str) -> None:
        store = self.runtime.get_registry("conversation")
        reset = getattr(store, "reset", None) or getattr(store, "clear", None)
        if reset is not None:
            reset(session_id)

    def get_session_info(self, session_id: str):
        store = self.runtime.get_registry("conversation")
        snapshot = getattr(store, "snapshot", None)
        if snapshot is not None:
            return snapshot(session_id)
        return None

    def shutdown(self) -> None:
        self.runtime.shutdown()


_DEFAULT_APP: CAIApp | None = None


def get_default_app() -> CAIApp:
    global _DEFAULT_APP
    if _DEFAULT_APP is None:
        _DEFAULT_APP = CAIApp(get_default_runtime())
    return _DEFAULT_APP


def set_default_app(app: CAIApp | None) -> None:
    global _DEFAULT_APP
    _DEFAULT_APP = app