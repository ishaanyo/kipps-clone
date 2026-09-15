from datetime import datetime
from loguru import logger

# Simple in-memory CRM for demo
_leads = {}


def update_crm(
    phone: str,
    name: str = None,
    email: str = None,
    company: str = None,
    status: str = "new",
    notes: str = None,
    score: int = None,
) -> str:
    """
    Create or update a lead in the CRM.
    """
    lead = _leads.get(phone, {"phone": phone, "created_at": datetime.utcnow().isoformat()})
    
    if name:
        lead["name"] = name
    if email:
        lead["email"] = email
    if company:
        lead["company"] = company
    if status:
        lead["status"] = status
    if notes:
        lead["notes"] = notes
    if score is not None:
        lead["score"] = score
    
    lead["updated_at"] = datetime.utcnow().isoformat()
    _leads[phone] = lead
    
    logger.success(f"CRM updated for {phone}: {lead}")
    return f"Lead updated successfully for {phone}. Current status: {lead.get('status', 'new')}"


def get_lead(phone: str) -> str:
    lead = _leads.get(phone)
    if not lead:
        return f"No lead found for {phone}"
    return str(lead)
