"""
Department-aware Knowledge Base with simple RAG (keyword + department routing).
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger
import re


class KnowledgeBase:
    def __init__(self):
        # department_name -> list of text chunks
        self.departments: Dict[str, List[str]] = {}
        self.general: List[str] = []
        self.department_aliases = {
            "personal loan": "personal_loan",
            "personal": "personal_loan",
            "pl": "personal_loan",
            "home loan": "home_loan",
            "home": "home_loan",
            "housing": "home_loan",
            "business loan": "business_loan",
            "business": "business_loan",
            "msme": "business_loan",
            "credit card": "credit_card",
            "card": "credit_card",
            "creditcard": "credit_card",
            "support": "customer_support",
            "customer support": "customer_support",
            "general": "customer_support",
            "help": "customer_support",
            "query": "customer_support",
            "collections": "collections",
            "recovery": "collections",
            "overdue": "collections",
            "emi issue": "collections",
        }

    def add_text(self, text: str, department: str = "general"):
        text = text.strip()
        if not text:
            return
        if department == "general":
            self.general.append(text)
        else:
            self.departments.setdefault(department, []).append(text)
        logger.info(f"Added knowledge to [{department}] ({len(text)} chars)")

    def add_file(self, path: str, department: Optional[str] = None):
        p = Path(path)
        if not p.exists():
            logger.warning(f"Knowledge file not found: {path}")
            return

        content = p.read_text(encoding="utf-8")
        # Auto-detect department from filename if not given
        if department is None:
            name = p.stem.lower()
            department = name if name in [
                "personal_loan", "home_loan", "business_loan",
                "credit_card", "customer_support", "collections"
            ] else "general"

        self.add_text(content, department=department)

    def load_departments_folder(self, folder: str = "knowledge_docs/departments"):
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Departments folder not found: {folder}")
            return
        for file in folder_path.glob("*.txt"):
            self.add_file(str(file))

    def detect_department(self, query: str) -> Optional[str]:
        """Return department key from user query, or None."""
        q = query.lower().strip()
        for alias, dept in self.department_aliases.items():
            if alias in q:
                return dept
        return None

    def list_departments(self) -> List[str]:
        return list(self.departments.keys())

    def get_department_display_name(self, key: str) -> str:
        mapping = {
            "personal_loan": "Personal Loan",
            "home_loan": "Home Loan",
            "business_loan": "Business Loan",
            "credit_card": "Credit Card",
            "customer_support": "Customer Support",
            "collections": "Collections / Recovery",
        }
        return mapping.get(key, key.replace("_", " ").title())

    def search(self, query: str, department: Optional[str] = None, top_k: int = 4) -> str:
        """
        Simple RAG: score chunks by keyword overlap.
        If department is given, search only that department + general.
        """
        candidates: List[Tuple[int, str, str]] = []  # score, text, dept

        def score_text(text: str, q: str) -> int:
            words = set(re.findall(r"\w+", q.lower()))
            text_lower = text.lower()
            return sum(1 for w in words if len(w) > 2 and w in text_lower)

        # Search specific department
        if department and department in self.departments:
            for chunk in self.departments[department]:
                s = score_text(chunk, query)
                if s > 0:
                    candidates.append((s, chunk, department))

        # Always include general
        for chunk in self.general:
            s = score_text(chunk, query)
            if s > 0:
                candidates.append((s, chunk, "general"))

        # If no department or weak results, search all
        if not candidates or (department and len(candidates) < 2):
            for dept, chunks in self.departments.items():
                if dept == department:
                    continue
                for chunk in chunks:
                    s = score_text(chunk, query)
                    if s > 0:
                        candidates.append((s, chunk, dept))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:top_k]

        if not top:
            return "No specific information found in the knowledge base."

        parts = []
        for score, text, dept in top:
            parts.append(f"[Source: {self.get_department_display_name(dept)}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def get_context_for_department(self, department: str, max_chars: int = 3500) -> str:
        chunks = self.departments.get(department, []) + self.general
        combined = "\n\n".join(chunks)
        if len(combined) > max_chars:
            return combined[:max_chars] + "\n...[truncated]"
        return combined

    def get_all_departments_summary(self) -> str:
        lines = ["Available Departments at SecureLoan Finance:"]
        for key in self.list_departments():
            lines.append(f"- {self.get_department_display_name(key)}")
        return "\n".join(lines)
