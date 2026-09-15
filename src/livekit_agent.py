"""
LiveKit voice agent worker — continuous call via WebRTC.
Uses AICredits for STT / LLM / TTS (OpenAI-compatible gateway).

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
    from livekit.agents import (
        Agent,
        AgentSession,
        JobContext,
        WorkerOptions,
        cli,
        ConversationItemAddedEvent,
        UserInputTranscribedEvent,
    )
    from livekit.plugins import openai, silero
except ImportError as e:
    print('Install: pip install "livekit-agents[openai,silero]" livekit-api livekit')
    raise SystemExit(1) from e

from src.utils.lead_store import LeadStore

AICREDITS_BASE = (os.getenv("AICREDITS_BASE_URL") or "https://api.aicredits.in/v1").rstrip("/")
AICREDITS_KEY = os.getenv("AICREDITS_API_KEY") or os.getenv("OPENAI_API_KEY")

INSTRUCTIONS = """
You are the SecureLoan Finance AI voice agent on a live phone-style call.
Speak naturally in short sentences (1-3). Prefer simple Hinglish or English matching the caller.
Departments: Personal Loan, Home Loan, Business Loan, Gold Loan, Credit Card, Customer Support, Collections.
If the department is unclear, ask which one they need.
Qualify loan buyers: property/amount, income, city. Offer callback when ready.
Never invent rates. Be polite. No markdown, asterisks, or special symbols — this is spoken aloud.
Always reply to what the user just said.
"""


def _build_session() -> AgentSession:
    if not AICREDITS_KEY:
        raise RuntimeError("AICREDITS_API_KEY required")

    logger.info(f"AICredits base={AICREDITS_BASE}")

    # STT: try whisper-1 (AICredits supports openai/whisper-1 and whisper-1)
    stt_model = os.getenv("LIVEKIT_STT_MODEL", "whisper-1")
    llm_model = os.getenv("LIVEKIT_LLM_MODEL", "gpt-4o-mini")
    tts_model = os.getenv("LIVEKIT_TTS_MODEL", "tts-1")
    tts_voice = os.getenv("LIVEKIT_TTS_VOICE", "alloy")

    stt_kwargs = dict(model=stt_model, api_key=AICREDITS_KEY)
    llm_kwargs = dict(model=llm_model, api_key=AICREDITS_KEY, temperature=0.55)
    tts_kwargs = dict(model=tts_model, voice=tts_voice, api_key=AICREDITS_KEY)

    # base_url support varies by plugin version
    for kwargs, base in (
        (stt_kwargs, AICREDITS_BASE),
        (llm_kwargs, AICREDITS_BASE),
        (tts_kwargs, AICREDITS_BASE),
    ):
        kwargs["base_url"] = base

    try:
        stt = openai.STT(**stt_kwargs)
    except TypeError:
        stt_kwargs.pop("base_url", None)
        stt = openai.STT(**stt_kwargs)
        logger.warning("STT without base_url — set OPENAI_API_KEY to AICredits key and hope default host works, or upgrade plugin")

    try:
        llm = openai.LLM(**llm_kwargs)
    except TypeError:
        llm_kwargs.pop("base_url", None)
        llm = openai.LLM(**llm_kwargs)

    try:
        tts = openai.TTS(**tts_kwargs)
    except TypeError:
        tts_kwargs.pop("base_url", None)
        tts = openai.TTS(**tts_kwargs)

    vad = silero.VAD.load()

    session_kwargs = dict(stt=stt, llm=llm, tts=tts, vad=vad)
    # Optional kwargs depending on livekit-agents version
    for k, v in (("allow_interruptions", True), ("min_endpointing_delay", 0.8)):
        session_kwargs[k] = v
    try:
        return AgentSession(**session_kwargs)
    except TypeError:
        session_kwargs.pop("allow_interruptions", None)
        session_kwargs.pop("min_endpointing_delay", None)
        return AgentSession(**session_kwargs)


async def entrypoint(ctx: JobContext):
    logger.info(f"Agent joining room: {ctx.room.name}")
    await ctx.connect()

    session = _build_session()
    agent = Agent(instructions=INSTRUCTIONS)

    lead_store = LeadStore()
    transcript_parts: list[dict] = []

    def _on_user_transcript(ev: UserInputTranscribedEvent):
        text = getattr(ev, "transcript", None) or ""
        is_final = getattr(ev, "is_final", True)
        logger.info(f"USER transcript (final={is_final}): {text!r}")
        if is_final and text.strip():
            transcript_parts.append({"role": "user", "content": text.strip()})

    def _on_item(ev: ConversationItemAddedEvent):
        try:
            item = ev.item
            role = getattr(item, "role", None)
            content = getattr(item, "text_content", None) or getattr(item, "content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            logger.info(f"ITEM role={role}: {str(content)[:120]!r}")
            if role and content:
                transcript_parts.append({"role": str(role), "content": str(content)})
        except Exception as e:
            logger.warning(f"item handler: {e}")

    session.on("user_input_transcribed", _on_user_transcript)
    session.on("conversation_item_added", _on_item)

    await session.start(room=ctx.room, agent=agent)

    await session.generate_reply(
        instructions=(
            "Greet the caller briefly in one or two short sentences. "
            "Welcome them to SecureLoan Finance and ask which department they need."
        )
    )

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
