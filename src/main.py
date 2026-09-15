#!/usr/bin/env python3
"""
Kipps.AI Clone – AI Voice Agent (powered by AICredits)

Usage:
    python -m src.main --mode text         # Text chat (recommended for testing)
    python -m src.main --mode mic          # Microphone conversation
    python -m src.main --mode server       # FastAPI server
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import setup_logger
from src.voice_agent import VoiceAgent
from loguru import logger


def run_mic_mode(agent: VoiceAgent):
    agent.run_conversation_loop(max_turns=12)


def run_text_mode(agent: VoiceAgent):
    print("\n" + "=" * 60)
    print(f"  {agent.config['agent']['name']} – Text Mode (AICredits)")
    print("  Type 'quit' or 'exit' to end")
    print("=" * 60 + "\n")

    greeting = agent.config["conversation"].get("greeting", "Hi! How can I help?")
    print(f"Agent: {greeting}")
    agent.messages.append({"role": "assistant", "content": greeting})

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("Agent: Thank you! Goodbye.")
            break

        if not user_input:
            continue

        response = agent.process_text(user_input)
        print(f"Agent: {response}")

    agent._save_transcript()


def run_server_mode(agent: VoiceAgent, port: int = 8000):
    from fastapi import FastAPI, UploadFile, File
    import uvicorn

    app = FastAPI(title="Kipps Clone Voice Agent (AICredits)", version="0.2.0")

    @app.get("/")
    def health():
        return {"status": "ok", "agent": agent.config["agent"]["name"], "provider": "aicredits"}

    @app.post("/chat")
    async def chat(payload: dict):
        text = payload.get("text", "")
        response = agent.process_text(text)
        return {"response": response}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        audio_bytes = await file.read()
        text = agent.stt.transcribe(audio_bytes, filename=file.filename or "audio.wav")
        return {"transcript": text}

    logger.info(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(description="Kipps.AI Clone – AI Voice Agent (AICredits)")
    parser.add_argument("--mode", choices=["mic", "text", "server"], default="text")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG_PATH", "config/agent_config.yaml"))
    args = parser.parse_args()

    setup_logger(os.getenv("LOG_LEVEL", "INFO"))
    logger.info("Starting Kipps Clone Voice Agent (AICredits)...")

    if not os.getenv("AICREDITS_API_KEY"):
        logger.error(
            "AICREDITS_API_KEY is required.\n"
            "1. Sign up at https://aicredits.in\n"
            "2. Top up via UPI\n"
            "3. Create API key in Dashboard → API Keys\n"
            "4. Put it in .env"
        )
        sys.exit(1)

    agent = VoiceAgent(config_path=args.config)

    if args.mode == "mic":
        run_mic_mode(agent)
    elif args.mode == "text":
        run_text_mode(agent)
    elif args.mode == "server":
        run_server_mode(agent, port=args.port)


if __name__ == "__main__":
    main()
