"""REST API for multi-company Voice Agent SaaS dashboard."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Any

from .store import TenantStore
from .models import AgentConfig

router = APIRouter(prefix="/api/saas", tags=["saas"])
store = TenantStore()


class CompanyCreate(BaseModel):
    name: str
    slug: str
    contact_email: str = ""
    plan: str = "starter"


class AgentCreate(BaseModel):
    name: str
    slug: str = "main"
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None
    config: dict[str, Any] | None = None


def _auth_company(x_api_key: str | None) -> str | None:
    """Optional company API key → company_id."""
    if not x_api_key:
        return None
    for co in store.list_companies():
        if co.api_key == x_api_key:
            return co.id
    return None


@router.get("/health")
def saas_health():
    return {
        "ok": True,
        "companies": len(store.list_companies()),
        "agents": len(store.list_agents()),
    }


@router.get("/companies")
def list_companies():
    return [c.to_dict() for c in store.list_companies()]


@router.post("/companies")
def create_company(body: CompanyCreate):
    try:
        co = store.create_company(body.name, body.slug, body.contact_email, body.plan)
        return co.to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/companies/{company_id}")
def get_company(company_id: str):
    co = store.get_company(company_id) or store.get_company_by_slug(company_id)
    if not co:
        raise HTTPException(404, "Company not found")
    agents = [a.to_dict() for a in store.list_agents(co.id)]
    return {"company": co.to_dict(), "agents": agents}


@router.get("/companies/{company_id}/agents")
def list_company_agents(company_id: str):
    co = store.get_company(company_id) or store.get_company_by_slug(company_id)
    if not co:
        raise HTTPException(404, "Company not found")
    return [a.to_dict() for a in store.list_agents(co.id)]


@router.post("/companies/{company_id}/agents")
def create_agent(company_id: str, body: AgentCreate):
    co = store.get_company(company_id) or store.get_company_by_slug(company_id)
    if not co:
        raise HTTPException(404, "Company not found")
    try:
        ag = store.create_agent(co.id, body.name, body.slug, body.config)
        if body.description:
            ag = store.update_agent(ag.id, description=body.description)
        return ag.to_dict()
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@router.get("/agents/{agent_id}")
def get_agent(agent_id: str):
    ag = store.get_agent(agent_id)
    if not ag:
        raise HTTPException(404, "Agent not found")
    return ag.to_dict()


@router.patch("/agents/{agent_id}")
def update_agent(agent_id: str, body: AgentUpdate):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return store.update_agent(agent_id, **fields).to_dict()
    except KeyError:
        raise HTTPException(404, "Agent not found")


@router.get("/resolve/{company_slug}")
@router.get("/resolve/{company_slug}/{agent_slug}")
def resolve_call(company_slug: str, agent_slug: str = "main"):
    """Public: resolve company+agent for browser/phone call widget."""
    resolved = store.resolve_for_call(company_slug, agent_slug)
    if not resolved:
        raise HTTPException(404, "Agent not found or inactive")
    co, ag = resolved
    return {
        "company": {"id": co.id, "name": co.name, "slug": co.slug},
        "agent": {
            "id": ag.id,
            "name": ag.name,
            "slug": ag.slug,
            "greeting": ag.config.greeting,
            "language": ag.config.language,
            "tts_voice": ag.config.tts_voice,
        },
    }
