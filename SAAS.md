# Multi-Company AI Voice Agent SaaS

Architecture inspired by [Kipps.AI](https://www.kipps.ai/) — many companies, each with one or more voice agents.

## What you get

| Feature | Status |
|---------|--------|
| Multi-tenant companies | ✅ |
| Per-company voice agents | ✅ |
| Per-agent prompt / voice / knowledge | ✅ |
| Dashboard UI | ✅ `/dashboard` |
| REST API | ✅ `/api/saas/*` |
| Browser call widget | ✅ `/?company=slug` |
| LiveKit worker loads tenant config | ✅ from room name |
| Billing / SSO / phone numbers | 🔜 next |

## Quick start

```bash
# Terminal 1 – worker
python -m src.livekit_agent dev

# Terminal 2 – SaaS server
python -m src.call_server
```

- Dashboard: http://127.0.0.1:8080/dashboard  
- Call SecureLoan: http://127.0.0.1:8080/?company=secureloan  
- Call MediCare: http://127.0.0.1:8080/?company=medicare  
- Call EduSpark: http://127.0.0.1:8080/?company=eduspark  

## API examples

```bash
# List companies
curl http://127.0.0.1:8080/api/saas/companies

# Create company
curl -X POST http://127.0.0.1:8080/api/saas/companies \
  -H 'Content-Type: application/json' \
  -d '{"name":"My Biz","slug":"mybiz","plan":"starter"}'

# Create agent
curl -X POST http://127.0.0.1:8080/api/saas/companies/CO_ID/agents \
  -H 'Content-Type: application/json' \
  -d '{"name":"Sales Agent","slug":"sales","config":{"greeting":"Hi from My Biz!","tts_voice":"priya"}}'

# Resolve for call widget
curl http://127.0.0.1:8080/api/saas/resolve/secureloan/main
```

## Data

Tenants are stored in `data/tenants.json` (JSON file store).  
Swap `TenantStore` for Postgres when you scale.

## Roadmap (Kipps-like)

1. Phone numbers (Twilio / Exotel / Plivo) per agent  
2. WhatsApp + web chat agents  
3. Visual flow builder  
4. Usage metering & plans  
5. Team seats + SSO  
6. CRM / calendar integrations per company  
