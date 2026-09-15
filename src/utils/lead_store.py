"""
Append-only storage for conversation leads / call summaries.
Format: JSON Lines (.jsonl) — one JSON object per line (easy to append & process).
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


class LeadStore:
    def __init__(self, path: str = "data/leads.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save_conversation(
        self,
        messages: List[Dict],
        department: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Extract useful info from the conversation and append one record.
        Returns the path of the file written to.
        """
        # Extract basic fields from messages
        user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        assistant_texts = [m.get("content", "") for m in messages if m.get("role") == "assistant"]

        # Try to pull name / phone from conversation (simple heuristics)
        name = None
        phone = None
        for text in user_texts:
            lower = text.lower()
            # very light extraction
            if "my name is" in lower:
                name = text.split("is")[-1].strip(" .,").title()
            if any(c.isdigit() for c in text) and len([c for c in text if c.isdigit()]) >= 10:
                digits = "".join(c for c in text if c.isdigit())
                if len(digits) >= 10:
                    phone = digits[-10:]  # last 10 digits

        record = {
            "id": datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "department": department or "unknown",
            "name": name,
            "phone": phone,
            "turn_count": len([m for m in messages if m.get("role") == "user"]),
            "user_messages": user_texts,
            "assistant_messages": assistant_texts,
            "full_transcript": [
                {"role": m.get("role"), "content": m.get("content")}
                for m in messages
                if m.get("content")
            ],
            "extra": extra or {},
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.success(f"Lead/conversation saved → {self.path} (id={record['id']})")
        return str(self.path)

    def load_all(self) -> List[Dict]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
