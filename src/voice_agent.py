import os
import yaml
from typing import List, Dict, Optional
from loguru import logger

from src.llm.openai_llm import OpenAILLM
from src.stt.openai_stt import OpenAISTT
from src.tts.openai_tts import OpenAITTS
from src.knowledge.knowledge_base import KnowledgeBase
from src.tools.calendar import book_appointment
from src.tools.crm import update_crm
from src.utils.audio import record_audio, play_audio, numpy_to_wav_bytes
from src.utils.lead_store import LeadStore
import numpy as np


class VoiceAgent:
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        self.config = self._load_config(config_path)
        self.messages: List[Dict] = []
        self.kb = KnowledgeBase()
        self.current_department: Optional[str] = None   # tracks selected department
        self._setup_knowledge()
        self._setup_llm()
        self._setup_stt()
        self._setup_tts()
        self.turn_count = 0
        self.lead_store = LeadStore()
        logger.info(f"Voice Agent '{self.config['agent']['name']}' ready (AICredits)")

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _setup_knowledge(self):
        # Load general sources
        for source in self.config.get("knowledge", {}).get("sources", []):
            if source["type"] == "text":
                self.kb.add_text(source["content"], department="general")
            elif source["type"] == "file":
                self.kb.add_file(source["path"])

        # Load all department files
        if self.config.get("knowledge", {}).get("load_departments_folder", True):
            self.kb.load_departments_folder("knowledge_docs/departments")

        logger.info(f"Departments loaded: {self.kb.list_departments()}")

    def _setup_llm(self):
        llm_cfg = self.config["agent"]["llm"]
        self.llm = OpenAILLM(
            model=llm_cfg.get("model", "openai/gpt-4o-mini"),
            temperature=llm_cfg.get("temperature", 0.5),
            max_tokens=llm_cfg.get("max_tokens", 350),
        )

        self.llm.register_tool(
            name="book_appointment",
            description="Book a callback, demo or branch visit for the caller.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Caller's full name"},
                    "phone": {"type": "string", "description": "Caller's phone number"},
                    "datetime_str": {"type": "string", "description": "YYYY-MM-DD HH:MM"},
                    "purpose": {"type": "string", "description": "Purpose / department"},
                },
                "required": ["name", "phone", "datetime_str"],
            },
            handler=book_appointment,
        )

        self.llm.register_tool(
            name="update_crm",
            description="Save or update lead information in the CRM.",
            parameters={
                "type": "object",
                "properties": {
                    "phone": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "company": {"type": "string"},
                    "status": {"type": "string", "enum": ["new", "qualified", "not_interested", "booked"]},
                    "notes": {"type": "string"},
                    "score": {"type": "integer"},
                },
                "required": ["phone"],
            },
            handler=update_crm,
        )

        self.llm.register_tool(
            name="transfer_to_human",
            description="Transfer the call to a human agent or specific department.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "department": {"type": "string", "description": "Target department if known"},
                },
                "required": ["reason"],
            },
            handler=lambda reason, department=None: (
                f"Transferring you to {'the ' + department + ' team' if department else 'a human agent'}. "
                f"Reason: {reason}. Please hold."
            ),
        )

    def _setup_stt(self):
        stt_cfg = self.config["agent"].get("stt", {})
        self.stt = OpenAISTT(model=stt_cfg.get("model", "openai/whisper-1"))
        logger.info("STT → AICredits Whisper")

    def _setup_tts(self):
        tts_cfg = self.config["agent"].get("voice", {})
        self.tts = OpenAITTS(
            model=tts_cfg.get("model", "openai/tts-1"),
            voice=tts_cfg.get("voice", "alloy"),
        )
        logger.info(f"TTS → AICredits ({tts_cfg.get('voice', 'alloy')})")

    def _build_system_prompt(self, user_text: str = "") -> str:
        base = self.config["agent"]["system_prompt"]
        persona = self.config["agent"].get("persona", "")

        # Detect department from current message if not set
        detected = self.kb.detect_department(user_text)
        if detected and not self.current_department:
            self.current_department = detected
            logger.info(f"Department set to: {detected}")

        # Also try to detect from conversation history
        if not self.current_department:
            for msg in reversed(self.messages[-6:]):
                if msg.get("role") == "user":
                    detected = self.kb.detect_department(msg.get("content", ""))
                    if detected:
                        self.current_department = detected
                        break

        # Build RAG context
        if self.current_department:
            rag_context = self.kb.search(user_text or "loan", department=self.current_department)
            dept_name = self.kb.get_department_display_name(self.current_department)
            dept_info = f"\nCURRENT DEPARTMENT: {dept_name}\n\nRELEVANT KNOWLEDGE:\n{rag_context}"
        else:
            dept_info = f"\n{self.kb.get_all_departments_summary()}\n\nNo department selected yet. Ask the caller which department they need."

        return f"""{persona}

{base}
{dept_info}
"""

    def process_text(self, user_text: str) -> str:
        if not user_text.strip():
            return "I didn't catch that. Could you please repeat?"

        self.messages.append({"role": "user", "content": user_text})
        self.turn_count += 1

        # Update department if mentioned
        detected = self.kb.detect_department(user_text)
        if detected:
            self.current_department = detected
            logger.info(f"Department switched to: {detected}")

        system_prompt = self._build_system_prompt(user_text)
        result = self.llm.chat(self.messages, system_prompt=system_prompt)

        if result["tool_calls"]:
            tool_results = self.llm.execute_tools(result["tool_calls"])
            self.messages.append({
                "role": "assistant",
                "content": result["content"] or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": str(tc["arguments"]),
                        }
                    }
                    for tc in result["tool_calls"]
                ]
            })
            for tr in tool_results:
                self.messages.append(tr)
            result = self.llm.chat(self.messages, system_prompt=system_prompt)

        response_text = result["content"] or "Sorry, I couldn't generate a response."
        self.messages.append({"role": "assistant", "content": response_text})

        if len(self.messages) > 22:
            self.messages = self.messages[-18:]

        return response_text

    def speak(self, text: str):
        logger.info(f"Agent: {text}")
        audio = self.tts.synthesize_to_numpy(text)
        if len(audio) > 0:
            play_audio(audio, sample_rate=16000)
        else:
            logger.warning("No audio generated (check pydub/ffmpeg or TTS credits)")

    def listen(self, duration: float = 6.0) -> str:
        audio = record_audio(duration=duration)
        wav_bytes = numpy_to_wav_bytes(audio)
        return self.stt.transcribe(wav_bytes)

    def run_conversation_loop(self, max_turns: int = 15):
        greeting = self.config["conversation"].get(
            "greeting",
            "Welcome to SecureLoan Finance. Which department can I help you with today?"
        )
        print("\n" + "=" * 60)
        print(f"  {self.config['agent']['name']} is ready (powered by AICredits)")
        print("=" * 60 + "\n")

        self.speak(greeting)
        self.messages.append({"role": "assistant", "content": greeting})

        for i in range(max_turns):
            print(f"\n--- Turn {i + 1} ---")
            user_text = self.listen(duration=7.0)
            print(f"You: {user_text}")

            if not user_text:
                self.speak("I didn't hear anything. Are you still there?")
                continue

            end_phrases = self.config["conversation"].get("end_phrases", [])
            if any(p in user_text.lower() for p in end_phrases):
                self.speak("Thank you for calling SecureLoan Finance. Have a great day! Goodbye.")
                break

            response = self.process_text(user_text)
            self.speak(response)

        print("\nConversation ended.")
        self._save_transcript()

    def _save_transcript(self):
        os.makedirs("logs", exist_ok=True)
        path = f"logs/transcript_{self.turn_count}.txt"
        with open(path, "w") as f:
            f.write(f"Department: {self.current_department or 'Not selected'}\n\n")
            for m in self.messages:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                f.write(f"{role.upper()}: {content}\n\n")
        logger.info(f"Transcript saved → {path}")

        # Append structured lead data (JSONL) after every conversation
        try:
            self.lead_store.save_conversation(
                messages=self.messages,
                department=self.current_department,
            )
        except Exception as e:
            logger.error(f"Failed to save lead: {e}")
