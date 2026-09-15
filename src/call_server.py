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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from src.livekit_token import create_participant_token, _env

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
    url = _env("LIVEKIT_URL")
    key = _env("LIVEKIT_API_KEY")
    secret = _env("LIVEKIT_API_SECRET")
    return {
        "ok": True,
        "livekit_url": url[:40] + "..." if len(url) > 40 else url,
        "livekit_url_ok": url.startswith("wss://"),
        "api_key_prefix": key[:8] + "..." if key else None,
        "api_key_len": len(key),
        "api_secret_len": len(secret),
        "aicredits_set": bool(_env("AICREDITS_API_KEY")),
    }


@app.post("/api/token")
async def token(request: Request):
    """Issue a LiveKit token for the browser participant."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    room = payload.get("room") or f"secureloan-{uuid.uuid4().hex[:8]}"
    identity = payload.get("identity") or f"caller-{uuid.uuid4().hex[:6]}"

    url = _env("LIVEKIT_URL")
    if not url:
        raise HTTPException(status_code=500, detail="LIVEKIT_URL not set")
    if not url.startswith("wss://"):
        raise HTTPException(
            status_code=500,
            detail=f"LIVEKIT_URL must start with wss:// (got: {url[:30]}...)",
        )

    try:
        jwt = create_participant_token(identity=identity, room_name=room)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"token error: {e}") from e

    return {
        "token": jwt,
        "url": url,
        "room": room,
        "identity": identity,
    }


def main():
    missing = [k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not _env(k)]
    if missing:
        print("Missing env:", ", ".join(missing))
        print("Add them to .env then retry.")
        sys.exit(1)

    url = _env("LIVEKIT_URL")
    key = _env("LIVEKIT_API_KEY")
    secret = _env("LIVEKIT_API_SECRET")
    print("Call UI → http://127.0.0.1:8080")
    print(f"LIVEKIT_URL = {url}")
    print(f"API_KEY len = {len(key)}  prefix = {key[:10]}...")
    print(f"API_SECRET len = {len(secret)}")
    print("Also run:  python -m src.livekit_agent dev")
    print("Health check:  http://127.0.0.1:8080/api/health")
    uvicorn.run(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
