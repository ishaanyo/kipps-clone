"""
SecureLoan Voice Agent – GUI (voice-to-voice call interface)

Run:
    python -m src.gui_app
    python -m src.main --mode gui
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import numpy as np
from loguru import logger

from src.utils.logger import setup_logger
from src.voice_agent import VoiceAgent

setup_logger(os.getenv("LOG_LEVEL", "INFO"))

agent: VoiceAgent | None = None
call_active = False


def _tts_to_file(text: str) -> str | None:
    """Generate MP3 via AICredits and return file path for in-browser play (no ffmpeg)."""
    if not agent or not text:
        return None
    try:
        path = agent.tts.synthesize_to_file(text, suffix=".mp3")
        return path if path else None
    except Exception as e:
        logger.error(f"TTS file error: {e}")
        return None


def _format_chat(messages) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if role == "user":
            lines.append(f"**You:** {content}")
        elif role == "assistant" and content:
            lines.append(f"**Agent:** {content}")
        elif role == "tool" and content:
            lines.append(f"*System:* {content}")
    return "\n\n".join(lines) if lines else "_No messages yet_"


def start_call():
    global agent, call_active
    if not os.getenv("AICREDITS_API_KEY"):
        return (
            "❌ AICREDITS_API_KEY missing in .env",
            "🔴 Error",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    agent = VoiceAgent()
    call_active = True
    greeting = agent.config["conversation"].get(
        "greeting",
        "Namaste! SecureLoan Finance se baat ho rahi hai. Kaise madad kar sakta hoon?",
    )
    agent.messages.append({"role": "assistant", "content": greeting})
    audio_path = _tts_to_file(greeting)
    return (
        _format_chat(agent.messages),
        "🟢 Call connected — speak or type below",
        audio_path,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def end_call():
    global agent, call_active
    if agent is None:
        return (
            "No active call",
            "🔴 No active call",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    try:
        agent._save_transcript()
        saved = "✅ Call ended. Lead & transcript saved to `data/leads.jsonl` and `logs/`"
    except Exception as e:
        saved = f"⚠️ Call ended but save failed: {e}"
        logger.error(e)

    dept = agent.current_department or "unknown"
    turns = agent.turn_count
    summary = f"{saved}\n\n**Department:** {dept}  \n**Turns:** {turns}"

    agent = None
    call_active = False
    return (
        summary,
        "🔴 Call ended — data saved",
        None,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def process_text(user_text: str):
    global agent
    if not call_active or agent is None:
        return _format_chat([]), "Start a call first", None, user_text
    if not user_text or not user_text.strip():
        return _format_chat(agent.messages), "Type or speak something", None, ""

    response = agent.process_text(user_text.strip())
    audio_path = _tts_to_file(response)
    return _format_chat(agent.messages), "🟢 Listening…", audio_path, ""


def process_audio(audio):
    global agent
    if not call_active or agent is None:
        return _format_chat([]), "Start a call first", None
    if audio is None:
        return _format_chat(agent.messages), "No audio received", None

    try:
        sr, data = audio
        if data is None or len(data) == 0:
            return _format_chat(agent.messages), "Empty audio", None
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        peak = np.max(np.abs(data)) + 1e-8
        if peak > 1.0:
            data = data / peak

        import soundfile as sf
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, data, int(sr))
        with open(path, "rb") as f:
            wav_bytes = f.read()
        try:
            os.unlink(path)
        except Exception:
            pass

        text = agent.stt.transcribe(wav_bytes)
        if not text.strip():
            return _format_chat(agent.messages), "Didn't catch that — try again", None

        response = agent.process_text(text)
        audio_path = _tts_to_file(response)
        return _format_chat(agent.messages), f"You said: {text}", audio_path
    except Exception as e:
        logger.error(f"Audio process error: {e}")
        return _format_chat(agent.messages if agent else []), f"Error: {e}", None


def build_ui():
    with gr.Blocks(title="SecureLoan AI Voice Agent") as demo:
        gr.Markdown(
            """
            # 📞 SecureLoan AI Voice Agent
            Voice-to-voice call · Click **End Call** to save the lead automatically
            """
        )

        status = gr.Textbox(
            value="🔴 No active call — click Start Call",
            label="Call status",
            interactive=False,
        )

        chatbot = gr.Markdown(value="_Start a call to begin_")

        # Autoplay in browser — no download button emphasis
        agent_audio = gr.Audio(
            label="Agent voice",
            type="filepath",
            autoplay=True,
            interactive=False,
        )

        with gr.Row():
            start_btn = gr.Button("🟢 Start Call", variant="primary")
            end_btn = gr.Button("🔴 End Call", variant="stop", interactive=False)

        gr.Markdown("### Speak or type")
        mic = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Microphone — record, then stop to send",
            interactive=False,
        )
        with gr.Row():
            text_in = gr.Textbox(
                placeholder="Or type your message…",
                label="Text message",
                interactive=False,
                scale=4,
            )
            send_btn = gr.Button("Send", interactive=False, scale=1)

        gr.Markdown(
            "After **End Call**, data is saved to `data/leads.jsonl` and `logs/`."
        )

        start_btn.click(
            fn=start_call,
            outputs=[chatbot, status, agent_audio, mic, text_in, send_btn, start_btn, end_btn],
        )
        end_btn.click(
            fn=end_call,
            outputs=[chatbot, status, agent_audio, mic, text_in, send_btn, start_btn, end_btn],
        )
        send_btn.click(
            fn=process_text,
            inputs=[text_in],
            outputs=[chatbot, status, agent_audio, text_in],
        )
        text_in.submit(
            fn=process_text,
            inputs=[text_in],
            outputs=[chatbot, status, agent_audio, text_in],
        )
        mic.stop_recording(
            fn=process_audio,
            inputs=[mic],
            outputs=[chatbot, status, agent_audio],
        )

    return demo


def main():
    if not os.getenv("AICREDITS_API_KEY"):
        print("Set AICREDITS_API_KEY in .env first")
        sys.exit(1)
    demo = build_ui()
    launch_kwargs = dict(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
    try:
        launch_kwargs["theme"] = gr.themes.Soft(primary_hue="blue")
    except Exception:
        pass
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
