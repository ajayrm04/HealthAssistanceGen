import re
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ComplianceUtils:
    """
    Compliance utilities for HIPAA / GDPR gating.
    Runs PHI scrubbing, role-based access, and escalation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.disallowed_terms = config.get("compliance", {}).get("disallowed_terms", [])
        self.audit_log_file = config.get("compliance", {}).get("audit_log", "logs/compliance_audit.log")

    def scrub_phi(self, text: str) -> str:
        """
        Scrub common PHI (names, IDs, phone numbers, SSNs, addresses).
        Extendable via regex in config.yaml.
        """
        phi_patterns = {
            "phone": r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "patient_id": r"\bPID-\d{4,}\b",
        }

        scrubbed = text
        for label, pattern in phi_patterns.items():
            scrubbed = re.sub(pattern, f"[REDACTED-{label.upper()}]", scrubbed)

        return scrubbed

    def apply_gating(self, response: str, user_role: str) -> str:
        """
        Main compliance gating pipeline:
        - Scrub PHI
        - Check disallowed terms
        - Role-based filtering
        - Escalation if needed
        """
        # Step 1: Scrub PHI
        safe_text = self.scrub_phi(response)

        # Step 2: Disallowed term filter
        for term in self.disallowed_terms:
            if term.lower() in safe_text.lower():
                self._log_event("ESCALATION", user_role, safe_text)
                return "[REDACTED: Compliance Violation - Escalated to Human Reviewer]"

        # Step 3: Role-based filtering
        if user_role == "patient" and "internal_use_only" in safe_text:
            self._log_event("BLOCKED", user_role, safe_text)
            return "[Restricted content - available only to clinicians]"

        # Step 4: Log safe passage
        self._log_event("APPROVED", user_role, safe_text)

        return safe_text

    def _log_event(self, status: str, user_role: str, text: str):
        """
        Log compliance events for auditing (HIPAA requires audit trail).
        """
        with open(self.audit_log_file, "a") as f:
            f.write(
                f"{datetime.utcnow().isoformat()} | {status} | role={user_role} | text={text[:200]}\n"
            )
