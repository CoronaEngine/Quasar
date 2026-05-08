from __future__ import annotations

import argparse
import sys

from . import CAIApp, ChatRequest, StreamEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a CAI chat request from the command line.")
    parser.add_argument("text", nargs="*", help="Text prompt to send to CAI.")
    parser.add_argument("--session-id", default="cli", help="Session id used for the request.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw legacy chunks instead of decoded text/data events.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = " ".join(args.text).strip()
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        print("No prompt provided.", file=sys.stderr)
        return 2

    app = CAIApp()
    request = ChatRequest.from_text(text, session_id=args.session_id)

    try:
        for chunk in app.chat_stream(request):
            if args.raw:
                print(chunk)
                continue
            event = StreamEvent.from_legacy_chunk(chunk)
            if event.event_type == "data":
                print(event.to_legacy_chunk())
            elif event.event_type in {"done", "error", "cancelled"}:
                print(event.to_legacy_chunk())
    finally:
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
