from __future__ import annotations

from Quasar.cai import CAIApp, ChatRequest, StreamEvent


def main() -> None:
    app = CAIApp()
    request = ChatRequest.from_text("请用一句话介绍 CAI。", session_id="example-cli")

    try:
        for chunk in app.chat_stream(request):
            event = StreamEvent.from_legacy_chunk(chunk)
            print(event.to_dict())
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
