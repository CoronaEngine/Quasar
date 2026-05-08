# CoronaArtificialIntelligence

CoronaArtificialIntelligence, or CAI, is the AI runtime used by CabbageEditor. It now exposes a small public facade for host integration while keeping the legacy module loading system compatible.

The physical directory remains under CabbageEditor as a submodule. Standalone usage is supported through editable install or by importing the package from its parent directory.

## Install For Development

From this directory:

```powershell
python -m pip install -e .
```

Optional dependency groups:

```powershell
python -m pip install -e ".[langchain,workflow,media]"
python -m pip install -e ".[web]"
python -m pip install -e ".[object-recognition]"
```

`cabbage` is intentionally empty in this package because the CabbageEditor adapter lives beside the submodule in `plugins/AITool/cai_extensions`.

## Public Facade

```python
from CoronaArtificialIntelligence.cai import CAIApp, ChatRequest, StreamEvent

app = CAIApp()
request = ChatRequest.from_text("请总结这段文字", session_id="demo")

for chunk in app.chat_stream(request):
    event = StreamEvent.from_legacy_chunk(chunk)
    print(event.to_dict())
```

The facade currently wraps the legacy integrated stream handler. New host code should call `CAIApp`; old code can continue to use `ai_service.entrance.get_ai_entrance()`.

## Examples

- `examples/cli_chat.py`: minimal CLI-style script.
- `examples/fastapi_websocket.py`: FastAPI WebSocket integration sketch.
- `cai-chat`: console script installed by `pyproject.toml`.

## Architecture Notes

- `cai/`: public facade, runtime, protocol, and plugin manager.
- `ai_service/entrance.py`: legacy entrance compatibility layer.
- `ai_modules/`: feature modules loaded from `ai_service/module_settings.yaml`.
- `ai_tools/`: tool registry, response adapter, session helpers, and tool loading.
- `ai_workflow/`: workflow registries and LangGraph execution helpers.
- `ai_media_resource/`: media registry and storage adapters.
- `ai_agent/`: agent execution and conversation history.

## Documentation

See `docs/API_REFERENCE.md` for the public API surface and packaging notes.
