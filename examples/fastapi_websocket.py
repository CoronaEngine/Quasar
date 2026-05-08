from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from CoronaArtificialIntelligence.cai import CAIApp, ChatRequest, StreamEvent


app = FastAPI()
cai_app = CAIApp()


@app.websocket("/ai/chat")
async def chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        request_json = await websocket.receive_json()
        request = ChatRequest.from_any(request_json)
        for chunk in cai_app.chat_stream(request):
            await websocket.send_json(StreamEvent.from_legacy_chunk(chunk).to_dict())
    except WebSocketDisconnect:
        return


@app.on_event("shutdown")
def shutdown_cai() -> None:
    cai_app.shutdown()
