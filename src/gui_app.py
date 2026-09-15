"""
SecureLoan Voice Agent – GUI (voice-to-voice call interface)

Run:
    python -m src.gui_app
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
from src.utils.audio import numpy_to_wav_bytes

setup_logger(os.getenv("LOG_LEVEL", "INFO"))

# Global agent session (one call at a time)
agent: VoiceAgent | None = None
call_active = False


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
    status = "🟢 Call connected — speak or type below"
    return (
        _format_chat(agent.messages),
        status,
        audio_path,
        gr.update(interactive=True),   # mic
        gr.update(interactive=True),   # text
        gr.update(interactive=True),   # send
        gr.update(interactive=False),  # start
        gr.update(interactive=True),   # end
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
        saved = "✅ Call ended. Lead & transcript saved to data/leads.jsonl and logs/"
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
        gr.update(interactive=True),   # start
        gr.update(interactive=False),  # end
    )


def _tts_to_file(text: str) -> str | None:
    if not agent:
        return None
    try:
        audio = agent.tts.synthesize_to_numpy(text)
        if len(audio) == 0:
            # fallback: raw mp3 bytes
            mp3 = agent.tts.synthesize(text)
            if not mp3:
                return None
            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            with open(path, "wb") as f:
                f.write(mp3)
            return path
        # write wav
        import soundfile as sf
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio, 16000)
        return path
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
    """audio: (sample_rate, numpy array) from Gradio Microphone"""
    global agent
    if not call_active or agent is None:
        return _format_chat([]), "Start a call first", None
    if audio is None:
        return _format_chat(agent.messages), "No audio received", None

    try:
        sr, data = audio
        if data is None or len(data) == 0:
            return _format_chat(agent.messages), "Empty audio", None
        # mono float32
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.max() > 1.0 or data.min() < -1.0:
            data = data / (np.max(np.abs(data)) + 1e-8)
        # resample rough if needed (Whisper accepts various rates via wav)
        import soundfile as sf
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, data, sr)
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
    with gr.Blocks(
        title="SecureLoan AI Voice Agent",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .status-box { font-size: 1.1em; padding: 12px; border-radius: 8px; }
        """
    ) as demo:
        gr.Markdown(
            """
            # 📞 SecureLoan AI Voice Agent
            Voice-to-voice call · End call to save lead automatically
            """
        )

        status = gr.Textbox(
            value="🔴 No active call — click Start Call",
            label="Call status",
            interactive=False,
            elem_classes=["status-box"],
        )

        chatbot = gr.Markdown(value="_Start a call to begin_", label="Conversation")

        agent_audio = gr.Audio(label="Agent speaking", autoplay=True, type="filepath")

        with gr.Row():
            start_btn = gr.Button("🟢 Start Call", variant="primary", scale=1)
            end_btn = gr.Button("🔴 End Call", variant="stop", interactive=False, scale=1)

        gr.Markdown("### Speak or type")
        with gr.Row():
            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Hold / record then release to send",
                interactive=False,
            )
        with gr.Row():
            text_in = gr.Textbox(
                placeholder="Or type your message here…",
                label="Text message",
                interactive=False,
                scale=4,
            )
            send_btn = gr.Button("Send", interactive=False, scale=1)

        gr.Markdown(
            "*After **End Call**, lead data is appended to `data/leads.jsonl` and transcript to `logs/`.*"
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
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
