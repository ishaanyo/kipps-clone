"""JSON-backed tenant store (companies + voice agents). Swap for Postgres later."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from .models import (
    AgentConfig,
    Company,
    VoiceAgent,
    new_agent,
    new_company,
)


class TenantStore:
    def __init__(self, path: str | Path | None = None):
        root = Path(__file__).resolve().parents[2]
        self.path = Path(path) if path else root / "data" / "tenants.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"companies": {}, "agents": {}}
        self._load()
        if not self._data["companies"]:
            self._seed()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"TenantStore load failed: {e}")
                self._data = {"companies": {}, "agents": {}}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _seed(self) -> None:
        """Demo multi-company data like Kipps templates."""
        demos = [
            (
                new_company("SecureLoan Finance", "secureloan", "ops@secureloan.demo", "growth"),
                new_agent(
                    "",
                    "SecureLoan Voice Agent",
                    "main",
                    AgentConfig(
                        persona="Professional Indian loan call-center agent. Hinglish OK.",
                        system_prompt=(
                            "You are SecureLoan Finance's live phone AI agent for India.\n"
                            "Departments: Personal Loan, Home Loan, Business Loan, Gold Loan, Credit Card, Support.\n"
                            "Ask which product they need, qualify (amount, city, income), offer callback.\n"
                            "Keep answers 1-3 sentences. Match caller's language (Hindi/Hinglish/English)."
                        ),
                        greeting="SecureLoan Finance mein aapka swagat hai! Kaunsa loan ya service chahiye?",
                        tts_voice="priya",
                        language="hi-IN",
                        knowledge_text="Personal loan 10.5% p.a. onwards. Home loan 8.4%. Gold loan 9%.",
                    ),
                ),
            ),
            (
                new_company("MediCare Hospitals", "medicare", "desk@medicare.demo", "starter"),
                new_agent(
                    "",
                    "Hospital Receptionist",
                    "reception",
                    AgentConfig(
                        persona="Warm hospital front-desk agent.",
                        system_prompt=(
                            "You are the AI receptionist for MediCare Hospitals.\n"
                            "Help with appointments, departments, visiting hours, and billing queries.\n"
                            "Collect patient name and preferred time before booking."
                        ),
                        greeting="Welcome to MediCare Hospitals. How may I help you today?",
                        tts_voice="priya",
                        language="en-IN",
                        knowledge_text="OPD 9am-5pm. Emergency 24x7. Cardiology, Ortho, Pediatrics.",
                    ),
                ),
            ),
            (
                new_company("EduSpark Academy", "eduspark", "sales@eduspark.demo", "starter"),
                new_agent(
                    "",
                    "Admissions Counselor",
                    "admissions",
                    AgentConfig(
                        persona="Friendly EdTech admissions counselor.",
                        system_prompt=(
                            "You are EduSpark Academy's admissions AI.\n"
                            "Qualify leads: course interest, budget, start date. Book demo class."
                        ),
                        greeting="Hi! This is EduSpark Academy. Which course are you interested in?",
                        tts_voice="ishita",
                        language="en-IN",
                        knowledge_text="Courses: Data Science, Full Stack, Digital Marketing. Fees from 25k.",
                    ),
                ),
            ),
        ]
        with self._lock:
            for company, agent in demos:
                agent.company_id = company.id
                self._data["companies"][company.id] = company.to_dict()
                self._data["agents"][agent.id] = agent.to_dict()
            self._save()
        logger.info(f"TenantStore seeded {len(demos)} demo companies → {self.path}")

    # ── Companies ──────────────────────────────────────────────
    def list_companies(self) -> list[Company]:
        with self._lock:
            return [Company.from_dict(v) for v in self._data["companies"].values()]

    def get_company(self, company_id: str) -> Company | None:
        with self._lock:
            d = self._data["companies"].get(company_id)
            return Company.from_dict(d) if d else None

    def get_company_by_slug(self, slug: str) -> Company | None:
        with self._lock:
            for v in self._data["companies"].values():
                if v.get("slug") == slug:
                    return Company.from_dict(v)
        return None

    def create_company(self, name: str, slug: str, email: str = "", plan: str = "starter") -> Company:
        slug = slug.strip().lower().replace(" ", "-")
        if self.get_company_by_slug(slug):
            raise ValueError(f"Company slug already exists: {slug}")
        co = new_company(name, slug, email, plan)
        with self._lock:
            self._data["companies"][co.id] = co.to_dict()
            self._save()
        return co

    def update_company(self, company_id: str, **fields) -> Company:
        with self._lock:
            d = self._data["companies"].get(company_id)
            if not d:
                raise KeyError(company_id)
            for k, v in fields.items():
                if k in d and k not in ("id", "created_at"):
                    d[k] = v
            self._data["companies"][company_id] = d
            self._save()
            return Company.from_dict(d)

    # ── Agents ─────────────────────────────────────────────────
    def list_agents(self, company_id: str | None = None) -> list[VoiceAgent]:
        with self._lock:
            out = []
            for v in self._data["agents"].values():
                if company_id and v.get("company_id") != company_id:
                    continue
                out.append(VoiceAgent.from_dict(v))
            return out

    def get_agent(self, agent_id: str) -> VoiceAgent | None:
        with self._lock:
            d = self._data["agents"].get(agent_id)
            return VoiceAgent.from_dict(d) if d else None

    def get_agent_by_slug(self, company_slug: str, agent_slug: str) -> VoiceAgent | None:
        co = self.get_company_by_slug(company_slug)
        if not co:
            return None
        with self._lock:
            for v in self._data["agents"].values():
                if v.get("company_id") == co.id and v.get("slug") == agent_slug:
                    return VoiceAgent.from_dict(v)
        return None

    def create_agent(
        self,
        company_id: str,
        name: str,
        slug: str,
        config: dict | AgentConfig | None = None,
    ) -> VoiceAgent:
        if not self.get_company(company_id):
            raise KeyError(f"Company not found: {company_id}")
        slug = slug.strip().lower().replace(" ", "-")
        for a in self.list_agents(company_id):
            if a.slug == slug:
                raise ValueError(f"Agent slug exists in company: {slug}")
        cfg = config if isinstance(config, AgentConfig) else AgentConfig.from_dict(config)
        ag = new_agent(company_id, name, slug, cfg)
        with self._lock:
            self._data["agents"][ag.id] = ag.to_dict()
            self._save()
        return ag

    def update_agent(self, agent_id: str, **fields) -> VoiceAgent:
        with self._lock:
            d = self._data["agents"].get(agent_id)
            if not d:
                raise KeyError(agent_id)
            if "config" in fields and isinstance(fields["config"], dict):
                cfg = d.get("config") or {}
                cfg.update(fields["config"])
                d["config"] = cfg
                del fields["config"]
            for k, v in fields.items():
                if k in d and k not in ("id", "company_id", "created_at"):
                    d[k] = v
            d["updated_at"] = time.time()
            self._data["agents"][agent_id] = d
            self._save()
            return VoiceAgent.from_dict(d)

    def resolve_for_call(self, company_slug: str, agent_slug: str = "main") -> tuple[Company, VoiceAgent] | None:
        """Resolve tenant + agent for an inbound/browser call."""
        co = self.get_company_by_slug(company_slug)
        if not co or co.status != "active":
            return None
        ag = self.get_agent_by_slug(company_slug, agent_slug)
        if not ag:
            agents = self.list_agents(co.id)
            ag = agents[0] if agents else None
        if not ag or ag.status != "active":
            return None
        return co, ag
