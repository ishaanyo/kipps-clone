"""
Call demo server: serves UI + LiveKit join tokens.

Run:
    python -m src.call_server
Then open http://127.0.0.1:8080
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.livekit_token import create_participant_token

app = FastAPI(title="SecureLoan LiveKit Call Demo")

STATIC = Path(__file__).parent.parent / "static"
STATIC.mkdir(exist_ok=True)


@app.get("/")
def index():
    page = STATIC / "call.html"
    if not page.exists():
        return HTMLResponse("<h1>call.html missing</h1>", status_code=500)
    return FileResponse(page)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "livekit_url_set": bool(os.getenv("LIVEKIT_URL")),
        "aicredits_set": bool(os.getenv("AICREDITS_API_KEY")),
    }


@app.post("/api/token")
def token(payload: dict | None = None):
    """Issue a LiveKit token for the browser participant."""
    payload = payload or {}
    room = payload.get("room") or f"secureloan-{uuid.uuid4().hex[:8]}"
    identity = payload.get("identity") or f"caller-{uuid.uuid4().hex[:6]}"

    try:
        jwt = create_participant_token(identity=identity, room_name=room)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    url = os.getenv("LIVEKIT_URL")
    if not url:
        raise HTTPException(status_code=500, detail="LIVEKIT_URL not set")

    return {
        "token": jwt,
        "url": url,
        "room": room,
        "identity": identity,
    }


def main():
    missing = [k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.getenv(k)]
    if missing:
        print("Missing env:", ", ".join(missing))
        print("Add them to .env then retry.")
        sys.exit(1)
    print("Call UI → http://127.0.0.1:8080")
    print("Also run the agent worker:  python -m src.livekit_agent dev")
    uvicorn.run(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
