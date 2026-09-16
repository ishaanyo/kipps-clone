"""Multi-tenant data models for the Voice Agent SaaS."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import time
import uuid


def _id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


@dataclass
class AgentConfig:
    """Per-agent runtime config (prompt, voice, models)."""
    persona: str = "You are a helpful AI voice assistant."
    system_prompt: str = "Help the caller professionally. Keep answers short for speech."
    language: str = "hi-IN"  # hi-IN | en-IN | auto
    tts_model: str = "sarvam/bulbul-v3"
    tts_voice: str = "priya"
    stt_model: str = "whisper-1"
    llm_model: str = "gpt-4o-mini"
    greeting: str = "Hello! How can I help you today?"
    knowledge_text: str = ""
    tools_enabled: list[str] = field(default_factory=lambda: ["book_appointment", "update_crm", "transfer_to_human"])
    max_call_seconds: int = 600

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "AgentConfig":
        d = d or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class VoiceAgent:
    id: str
    company_id: str
    name: str
    slug: str  # unique within company, used in URLs
    description: str = ""
    status: str = "active"  # active | paused | draft
    config: AgentConfig = field(default_factory=AgentConfig)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VoiceAgent":
        cfg = AgentConfig.from_dict(d.get("config"))
        return cls(
            id=d["id"],
            company_id=d["company_id"],
            name=d["name"],
            slug=d["slug"],
            description=d.get("description", ""),
            status=d.get("status", "active"),
            config=cfg,
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )


@dataclass
class Company:
    id: str
    name: str
    slug: str  # unique global, e.g. secureloan
    plan: str = "starter"  # starter | growth | enterprise
    status: str = "active"
    contact_email: str = ""
    api_key: str = field(default_factory=lambda: _id("co_key_"))
    created_at: float = field(default_factory=time.time)
    # billing placeholders
    monthly_minutes_limit: int = 1000
    monthly_minutes_used: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Company":
        return cls(
            id=d["id"],
            name=d["name"],
            slug=d["slug"],
            plan=d.get("plan", "starter"),
            status=d.get("status", "active"),
            contact_email=d.get("contact_email", ""),
            api_key=d.get("api_key") or _id("co_key_"),
            created_at=d.get("created_at", time.time()),
            monthly_minutes_limit=d.get("monthly_minutes_limit", 1000),
            monthly_minutes_used=d.get("monthly_minutes_used", 0.0),
        )


def new_company(name: str, slug: str, email: str = "", plan: str = "starter") -> Company:
    return Company(id=_id("co_"), name=name, slug=slug, contact_email=email, plan=plan)


def new_agent(company_id: str, name: str, slug: str, config: AgentConfig | None = None) -> VoiceAgent:
    return VoiceAgent(
        id=_id("ag_"),
        company_id=company_id,
        name=name,
        slug=slug,
        config=config or AgentConfig(),
    )
