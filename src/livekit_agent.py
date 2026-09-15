"""
LiveKit voice agent worker — continuous call via WebRTC.
Uses AICredits for STT / LLM / TTS (OpenAI-compatible gateway).

Run (dev):
    python -m src.livekit_agent dev

Run (prod start):
    python -m src.livekit_agent start
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Map AICredits key so OpenAI plugins pick it up if they only read OPENAI_API_KEY
if os.getenv("AICREDITS_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("AICREDITS_API_KEY")

from loguru import logger

try:
    from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
    from livekit.plugins import openai, silero
    try:
        from livekit.agents import RoomInputOptions
    except ImportError:
        RoomInputOptions = None
except ImportError as e:
    print("Install LiveKit agents:\n  pip install \"livekit-agents[openai,silero]\" livekit-api")
    raise SystemExit(1) from e

from src.utils.lead_store import LeadStore

AICREDITS_BASE = os.getenv("AICREDITS_BASE_URL", "https://api.aicredits.in/v1")
AICREDITS_KEY = os.getenv("AICREDITS_API_KEY") or os.getenv("OPENAI_API_KEY")

INSTRUCTIONS = """
You are the SecureLoan Finance AI voice agent on a live phone-style call.
Speak naturally in short sentences (1-3). Prefer simple Hinglish or English matching the caller.
Departments: Personal Loan, Home Loan, Business Loan, Gold Loan, Credit Card, Customer Support, Collections.
First ask which department they need if unclear.
Qualify home/loan buyers: property, amount, income, city. Offer callback when ready.
Never invent rates. Be polite. No markdown or special symbols in speech.
"""


def _build_session() -> AgentSession:
    if not AICREDITS_KEY:
        raise RuntimeError("AICREDITS_API_KEY required")

    # OpenAI-compatible plugins → AICredits
    stt = openai.STT(
        model=os.getenv("LIVEKIT_STT_MODEL", "whisper-1"),
        base_url=AICREDITS_BASE,
        api_key=AICREDITS_KEY,
    )
    llm = openai.LLM(
        model=os.getenv("LIVEKIT_LLM_MODEL", "gpt-4o-mini"),
        base_url=AICREDITS_BASE,
        api_key=AICREDITS_KEY,
        temperature=0.55,
    )
    tts = openai.TTS(
        model=os.getenv("LIVEKIT_TTS_MODEL", "tts-1"),
        voice=os.getenv("LIVEKIT_TTS_VOICE", "alloy"),
        base_url=AICREDITS_BASE,
        api_key=AICREDITS_KEY,
    )
    vad = silero.VAD.load()

    return AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
    )


async def entrypoint(ctx: JobContext):
    logger.info(f"Agent joining room: {ctx.room.name}")
    await ctx.connect()

    session = _build_session()
    agent = Agent(instructions=INSTRUCTIONS)

    lead_store = LeadStore()
    transcript_parts: list[dict] = []

    @session.on("user_input_transcribed")
    def on_user_transcript(ev):
        text = getattr(ev, "transcript", None) or getattr(ev, "text", None) or str(ev)
        if text:
            transcript_parts.append({"role": "user", "content": str(text)})

    @session.on("conversation_item_added")
    def on_item(ev):
        try:
            item = getattr(ev, "item", ev)
            role = getattr(item, "role", None)
            content = getattr(item, "text_content", None) or getattr(item, "content", None)
            if role and content:
                transcript_parts.append({"role": str(role), "content": str(content)})
        except Exception:
            pass

    start_kwargs = {"room": ctx.room, "agent": agent}
    if RoomInputOptions is not None:
        try:
            start_kwargs["room_input_options"] = RoomInputOptions()
        except Exception:
            pass
    await session.start(**start_kwargs)

    await session.generate_reply(
        instructions="Greet the caller briefly. Welcome them to SecureLoan Finance and ask which department they need: Personal Loan, Home Loan, Business Loan, Gold Loan, Credit Card, Support, or Collections."
    )

    # When participant disconnects, save lead
    @ctx.room.on("participant_disconnected")
    def on_disconnect(participant):
        try:
            if transcript_parts:
                lead_store.save_conversation(
                    messages=transcript_parts,
                    department="livekit_call",
                    extra={"room": ctx.room.name, "participant": participant.identity},
                )
                logger.success("Lead saved after disconnect")
        except Exception as e:
            logger.error(f"Lead save failed: {e}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
