# Kipps.AI Clone – AI Voice Agent (AICredits)

Production-style **AI Voice Agent** inspired by [Kipps.AI](https://www.kipps.ai/), powered **entirely by [AICredits](https://aicredits.in)**.

One API key → LLM + STT (Whisper) + TTS. Pay in ₹ via UPI. No international card needed.

## Features

- Human-like multi-turn voice conversations
- Tool calling (calendar booking, CRM update, human handoff)
- Knowledge base from text / files
- Configurable persona & system prompt
- Call recording + transcript logging
- Text mode, microphone mode, FastAPI server

## Quick Start

### 1. Get AICredits key
1. Go to https://aicredits.in and sign up
2. Top up wallet via UPI (min ₹50)
3. Dashboard → API Keys → Create key

### 2. Install

```bash
cd kipps-clone
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env → put your AICREDITS_API_KEY
```

### 3. Run

```bash
# Text mode (easiest)
python -m src.main --mode text

# Microphone
python -m src.main --mode mic

# API server
python -m src.main --mode server --port 8000
```

## Configuration

Edit `config/agent_config.yaml`:

| Setting | Default | Notes |
|---------|---------|-------|
| `llm.model` | `openai/gpt-4o-mini` | Any AICredits model (`openai/gpt-4o`, `anthropic/claude-...`, `google/gemini-...`) |
| `stt.model` | `openai/whisper-1` | Also `sarvam/saarika-v2` |
| `voice.model` | `openai/tts-1` | `openai/tts-1-hd` or `sarvam/bulbul-v2` |
| `voice.voice` | `alloy` | alloy / echo / fable / onyx / nova / shimmer |

## Project Structure

```
kipps-clone/
├── config/agent_config.yaml
├── knowledge_docs/
├── src/
│   ├── aicredits_client.py   ← single shared client
│   ├── voice_agent.py
│   ├── main.py
│   ├── llm/
│   ├── stt/
│   ├── tts/
│   ├── tools/
│   └── knowledge/
└── tests/
```

## How AICredits is used

| Component | Endpoint | Model example |
|-----------|----------|---------------|
| LLM | `/v1/chat/completions` | `openai/gpt-4o-mini` |
| STT | `/v1/audio/transcriptions` | `openai/whisper-1` |
| TTS | `/v1/audio/speech` | `openai/tts-1` |

All traffic goes through `https://api.aicredits.in/v1` with your single key.

## Notes

- For TTS playback you need `ffmpeg` installed (`sudo apt install ffmpeg` or equivalent).
- Credits are prepaid in INR — check balance with `GET /v1/credits`.
- Tool calling works the same as native OpenAI.

## License

MIT (educational / internal use). Do not use the name "Kipps" commercially.
