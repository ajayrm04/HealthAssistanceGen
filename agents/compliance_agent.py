# agents/compliance_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent
import re, json
from pathlib import Path
from services.phi_utils import detect_phi, redact_phi

ESCALATION_LOG = Path("data/escalations.log")
ESCALATION_LOG.parent.mkdir(parents=True, exist_ok=True)

# Basic policy rules - extend these in production
POLICY_BLOCK_PHRASES = [
    "prescribe", "dosage", "dose", "administer", "stop medication", "do not", "guarantee", "definitely"
]
SENSITIVE_PATTERNS = [
    r"\bssn\b", r"\bsocial security\b", r"\bcard number\b"
]

class ComplianceAgent(BaseAgent):
    def __init__(self, human_contact: str = None):
        super().__init__("compliance")
        self.human_contact = human_contact

    def check_policies(self, text: str) -> Dict[str,Any]:
        issues = []
        lower = text.lower()
        for p in POLICY_BLOCK_PHRASES:
            if p in lower:
                issues.append(f"policy_phrase:{p}")
        for pat in SENSITIVE_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                issues.append("phi_exposure")
        return {"issues": issues}

    def escalate(self, context: Dict[str,Any]):
        # Log to a file and optionally notify (placeholder)
        ESCALATION_LOG.write_text(ESCALATION_LOG.read_text() + "\n\n" + json.dumps(context, ensure_ascii=False))
        # In production: send to reviewer queue, email, or ticketing system
        return {"status":"escalated","note":"logged"}

    async def handle(self, state: Dict[str,Any]) -> Dict[str,Any]:
        draft = state.get("draft") or state.get("final_answer") or ""
        issues = self.check_policies(draft)
        if issues["issues"]:
            ctx = {"thread_id": state.get("thread_id"), "query": state.get("query"), "draft": draft, "issues": issues}
            esc = self.escalate(ctx)
            return {"type":"escalated", "issues": issues["issues"], "escalation": esc}
        # Approved - add mandatory safe phrasing/disclaimer for medical assistance
        safe = draft + "\n\nDisclaimer: This is not a definitive medical diagnosis. See a clinician for confirmation."
        return {"type":"approved", "final": safe}
    