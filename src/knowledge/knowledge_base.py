from typing import List, Optional
from pathlib import Path
from loguru import logger


class KnowledgeBase:
    def __init__(self):
        self.documents: List[str] = []

    def add_text(self, text: str):
        self.documents.append(text.strip())
        logger.info(f"Added text knowledge ({len(text)} chars)")

    def add_file(self, path: str):
        p = Path(path)
        if p.exists():
            content = p.read_text(encoding="utf-8")
            self.documents.append(content)
            logger.info(f"Loaded knowledge from {path}")
        else:
            logger.warning(f"Knowledge file not found: {path}")

    def get_context(self, max_chars: int = 3000) -> str:
        """Return combined knowledge as context string."""
        combined = "\n\n---\n\n".join(self.documents)
        if len(combined) > max_chars:
            return combined[:max_chars] + "\n...[truncated]"
        return combined

    def search(self, query: str, top_k: int = 3) -> str:
        """Very simple keyword search (replace with embeddings in production)."""
        query_lower = query.lower()
        scored = []
        for doc in self.documents:
            score = sum(1 for word in query_lower.split() if word in doc.lower())
            if score > 0:
                scored.append((score, doc))
        scored.sort(reverse=True)
        results = [doc for _, doc in scored[:top_k]]
        return "\n\n".join(results) if results else "No relevant information found."
