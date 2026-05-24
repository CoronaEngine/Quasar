# CAI API Reference

This reference describes the public facade added for standalone and host-embedded use. The legacy `ai_service.entrance` module remains available for compatibility, but new host integrations should prefer `Quasar.cai`.

## Import

```python
from Quasar.cai import CAIApp, CAIRuntime, ChatRequest, StreamEvent
```

When CAI is used inside CabbageEditor, AITool imports the same facade through the editor package path and installs `cai_extensions` into the current `CAIApp`.

## CAIApp

`CAIApp(runtime=None)` is the host-facing facade.

Main methods:

- `chat_stream(request)`: accepts `ChatRequest` or a legacy payload `dict`; returns an iterator of legacy stream chunks.
- `chat(request)`: collects `chat_stream()` into a list.
- `register_tool(tool)`, `register_tools(tools)`: register tools into the runtime tool registry.
- `register_workflow(workflow)`: register a workflow into the runtime workflow registry.
- `register_plugin(plugin)`: install a `CAIPlugin` into the runtime.
- `reset_session(session_id)`: clears session state when supported by the conversation store.
- `get_session_info(session_id)`: returns a session snapshot when supported.
- `shutdown()`: shuts down registered plugins.

## CAIRuntime

`CAIRuntime` owns runtime metadata, capabilities, entrance handlers, plugin manager, and lazy registry references.

Useful methods:

- `get_registry(name)`: resolve a registry such as `tool`, `workflow`, `media`, `conversation`, or `model`.
- `set_registry(name, registry)`: override a registry for tests or host integration.
- `set_capability(name, value)` and `get_capability(name)`: attach host capabilities to the runtime.
- `register_tool_loader_registrar(registrar)`: attach host tool loader registration callbacks.
- `register_entrance_handler(name, handler)`: provide a handler used by legacy-compatible calls.

## ChatRequest

`ChatRequest` is the minimal request object used by the facade.

```python
request = ChatRequest.from_text(
    "请总结这段文字",
    session_id="demo",
    metadata={"request_id": "req_demo"},
)
```

It can also wrap an existing legacy payload:

```python
request = ChatRequest.from_any(payload)
legacy_payload = request.to_legacy_payload()
```

## StreamEvent

`StreamEvent` normalizes legacy stream chunks into event-like objects:

```python
event = StreamEvent.from_legacy_chunk(chunk)
print(event.event_type, event.session_id, event.metadata)
```

Current event detection recognizes `data`, `heartbeat`, `done`, and `error` from legacy chunk fields.

## Plugins

Plugins implement the minimal `CAIPlugin` protocol:

```python
class MyPlugin:
    name = "my.plugin"
    enabled = True

    def register(self, runtime):
        runtime.set_capability("my_capability", object())
        return {"name": self.name}
```

Install with:

```python
app.register_plugin(MyPlugin())
```

## Packaging

From this directory:

```powershell
python -m pip install -e .
```

Then import with:

```python
from Quasar.cai import CAIApp
```
