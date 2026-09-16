"""Vercel serverless entry — FastAPI SaaS + dashboard. LiveKit worker runs elsewhere."""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if os.getenv("VERCEL") or os.getenv("VERCEL_ENV"):
    os.environ.setdefault("TENANT_STORE_PATH", "/tmp/tenants.json")
from src.call_server import app
__all__ = ["app"]
