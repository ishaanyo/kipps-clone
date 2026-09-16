"""
Multi-company Voice Agent SaaS server.

Run:
    python -m src.call_server
Then open:
    http://127.0.0.1:8080              → call widget
    http://127.0.0.1:8080/dashboard    → SaaS dashboard
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.livekit_token import create_participant_token, _env
from src.saas.api import router as saas_router
from src.saas.store import TenantStore

app = FastAPI(title="AI Voice Agent SaaS", version="0.2.0")
app.include_router(saas_router)

STATIC = Path(__file__).parent.parent / "static"
STATIC.mkdir(exist_ok=True)
store = TenantStore()


@app.get("/")
def index():
    page = STATIC / "call.html"
    if not page.exists():
        return HTMLResponse("<h1>call.html missing</h1>", status_code=500)
    return FileResponse(page)


@app.get("/dashboard")
@app.get("/dashboard/")
def dashboard():
    page = STATIC / "dashboard" / "index.html"
    if page.exists():
        return FileResponse(page)
    return HTMLResponse("<h1>Dashboard loading… open /api/saas/companies</h1>")


@app.get("/api/health")
def health():
    url = _env("LIVEKIT_URL")
    key = _env("LIVEKIT_API_KEY")
    secret = _env("LIVEKIT_API_SECRET")
    return {
        "ok": True,
        "saas": True,
        "companies": len(store.list_companies()),
        "agents": len(store.list_agents()),
        "livekit_url_ok": bool(url and url.startswith("wss://")),
        "api_key_len": len(key or ""),
        "aicredits_set": bool(_env("AICREDITS_API_KEY")),
        "sarvam_set": bool(_env("SARVAM_API_KEY")),
    }


@app.post("/api/token")
async def token(request: Request):
    """
    Issue LiveKit token. Body may include:
      company_slug, agent_slug, room, identity
    Room metadata carries tenant so the worker loads the right agent.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    company_slug = (payload.get("company_slug") or "secureloan").strip().lower()
    agent_slug = (payload.get("agent_slug") or "main").strip().lower()

    resolved = store.resolve_for_call(company_slug, agent_slug)
    if not resolved:
        # fallback: still allow call with default room naming
        company_name = company_slug
        agent_id = ""
        agent_name = "default"
        greeting = "Hello! How can I help you?"
    else:
        co, ag = resolved
        company_name = co.name
        agent_id = ag.id
        agent_name = ag.name
        greeting = ag.config.greeting
        company_slug = co.slug
        agent_slug = ag.slug

    room = payload.get("room") or f"{company_slug}-{uuid.uuid4().hex[:8]}"
    identity = payload.get("identity") or f"caller-{uuid.uuid4().hex[:6]}"

    url = _env("LIVEKIT_URL")
    if not url:
        raise HTTPException(status_code=500, detail="LIVEKIT_URL not set")
    if not url.startswith("wss://"):
        raise HTTPException(
            status_code=500,
            detail=f"LIVEKIT_URL must start with wss:// (got: {url[:30]}...)",
        )

    # Encode tenant in room name + return agent context to client
    try:
        jwt = create_participant_token(identity=identity, room_name=room)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"token error: {e}") from e

    return {
        "token": jwt,
        "url": url,
        "room": room,
        "identity": identity,
        "company_slug": company_slug,
        "agent_slug": agent_slug,
        "agent_id": agent_id,
        "company_name": company_name,
        "agent_name": agent_name,
        "greeting": greeting,
        # Worker reads this from job metadata if client passes it when creating room
        "agent_context": {
            "company_slug": company_slug,
            "agent_slug": agent_slug,
            "agent_id": agent_id,
        },
    }


def main():
    missing = [k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not _env(k)]
    if missing:
        print("Missing env:", ", ".join(missing))
        print("Add them to .env then retry.")
        sys.exit(1)

    print("══════════════════════════════════════════")
    print("  AI Voice Agent SaaS (multi-company)")
    print("══════════════════════════════════════════")
    print("Call UI     → http://127.0.0.1:8080")
    print("Dashboard   → http://127.0.0.1:8080/dashboard")
    print("SaaS API    → http://127.0.0.1:8080/api/saas/companies")
    print(f"Companies   → {len(store.list_companies())}  Agents → {len(store.list_agents())}")
    print("Also run:    python -m src.livekit_agent dev")
    print("══════════════════════════════════════════")
    uvicorn.run(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
