"""
LiveKit voice agent worker — continuous call via WebRTC.
STT/LLM/TTS via AICredits. Saves leads on hangup.

Run:
    python -m src.livekit_agent dev
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

if os.getenv("AICREDITS_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("AICREDITS_API_KEY")

from loguru import logger

try:
    from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
    from livekit.plugins import openai, silero
except ImportError as e:
    print('Install: pip install "livekit-agents[openai,silero]" livekit-api livekit httpx')
    raise SystemExit(1) from e

from src.utils.lead_store import LeadStore

AICREDITS_BASE = (os.getenv("AICREDITS_BASE_URL") or "https://api.aicredits.in/v1").rstrip("/")
AICREDITS_KEY = os.getenv("AICREDITS_API_KEY") or os.getenv("OPENAI_API_KEY")

INSTRUCTIONS = """
You are SecureLoan Finance's live phone AI agent in India.

LANGUAGE (critical):
- Detect the caller's language from what they say.
- Reply in the SAME language / mix they use.
- If they use Hindi or Hinglish, answer in natural Hinglish (spoken, not formal written Hindi).
- If pure English, use simple clear English.
- Examples of Hinglish style: "Bilkul, main help karta hoon.", "Aapko home loan chahiye kya?", "Achha, kitna loan chahiye roughly?"

BEHAVIOR:
- Short spoken replies (1-3 sentences). No markdown, bullets, or symbols.
- Departments: Personal Loan, Home Loan, Business Loan, Gold Loan, Credit Card, Customer Support, Collections.
- If department unclear, ask once which department.
- Qualify: amount, city, income/employment when relevant. Offer callback with name + phone.
- Never invent interest rates. If unsure, say you'll arrange a human callback.
- Always answer the user's last message — never stay silent after they speak.
"""


def _make_stt():
    """Prefer custom HTTP STT; fall back to openai plugin."""
    try:
        from src.aicredits_stt import AICreditsSTT
        stt = AICreditsSTT(
            model=os.getenv("LIVEKIT_STT_MODEL", "whisper-1"),
            base_url=AICREDITS_BASE,
            api_key=AICREDITS_KEY,
        )
        logger.info("Using AICredits HTTP STT")
        return stt
    except Exception as e:
        logger.warning(f"Custom STT unavailable ({e}), using openai.STT plugin")
        kwargs = dict(model=os.getenv("LIVEKIT_STT_MODEL", "whisper-1"), api_key=AICREDITS_KEY)
        try:
            return openai.STT(**kwargs, base_url=AICREDITS_BASE)
        except TypeError:
            return openai.STT(**kwargs)


def _make_llm():
    kwargs = dict(
        model=os.getenv("LIVEKIT_LLM_MODEL", "gpt-4o-mini"),
        api_key=AICREDITS_KEY,
        temperature=0.6,
    )
    try:
        return openai.LLM(**kwargs, base_url=AICREDITS_BASE)
    except TypeError:
        return openai.LLM(**kwargs)


def _make_tts():
    kwargs = dict(
        model=os.getenv("LIVEKIT_TTS_MODEL", "tts-1"),
        voice=os.getenv("LIVEKIT_TTS_VOICE", "alloy"),
        api_key=AICREDITS_KEY,
    )
    try:
        return openai.TTS(**kwargs, base_url=AICREDITS_BASE)
    except TypeError:
        return openai.TTS(**kwargs)


def _build_session() -> AgentSession:
    if not AICREDITS_KEY:
        raise RuntimeError("AICREDITS_API_KEY required")
    vad = silero.VAD.load()
    session_kwargs = dict(
        stt=_make_stt(),
        llm=_make_llm(),
        tts=_make_tts(),
        vad=vad,
    )
    try:
        return AgentSession(**session_kwargs, allow_interruptions=True, min_endpointing_delay=0.7)
    except TypeError:
        return AgentSession(**session_kwargs)


async def entrypoint(ctx: JobContext):
    logger.info(f"Agent joining room: {ctx.room.name}")
    await ctx.connect()

    session = _build_session()
    agent = Agent(instructions=INSTRUCTIONS)
    lead_store = LeadStore()
    transcript_parts: list[dict] = []

    def save_lead(reason: str = ""):
        try:
            if not transcript_parts:
                transcript_parts.append({
                    "role": "system",
                    "content": f"Call ended ({reason}) with no captured user text",
                })
            path = lead_store.save_conversation(
                messages=list(transcript_parts),
                department="livekit_call",
                extra={"room": ctx.room.name, "reason": reason},
            )
            logger.success(f"Lead saved ({reason}) → {path}")
        except Exception as e:
            logger.error(f"Lead save failed: {e}")

    @session.on("user_input_transcribed")
    def on_user(ev):
        text = getattr(ev, "transcript", None) or ""
        is_final = getattr(ev, "is_final", True)
        logger.info(f"USER STT final={is_final}: {text!r}")
        if is_final and str(text).strip():
            transcript_parts.append({"role": "user", "content": str(text).strip()})

    @session.on("conversation_item_added")
    def on_item(ev):
        try:
            item = ev.item
            role = getattr(item, "role", None)
            content = getattr(item, "text_content", None) or getattr(item, "content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            content = str(content or "").strip()
            if role and content:
                logger.info(f"ITEM {role}: {content[:100]!r}")
                transcript_parts.append({"role": str(role), "content": content})
        except Exception as e:
            logger.warning(f"item: {e}")

    @session.on("close")
    def on_close(ev=None):
        save_lead("session_close")

    await session.start(room=ctx.room, agent=agent)

    await session.generate_reply(
        instructions=(
            "Greet briefly in friendly Hinglish-English mix, like a real Indian call center agent. "
            "Welcome to SecureLoan Finance and ask which department they need. "
            "Keep it to two short spoken sentences."
        )
    )

    @ctx.room.on("participant_disconnected")
    def on_disconnect(participant):
        logger.info(f"Participant left: {participant.identity}")
        save_lead("participant_disconnected")

    @ctx.room.on("disconnected")
    def on_room_disconnected():
        save_lead("room_disconnected")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
