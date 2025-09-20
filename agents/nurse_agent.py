# agents/nurse_agent.py
from typing import Dict, Any
from services.slot_extractor import SlotExtractor, REQUIRED_SLOTS
from agents.base_agent import BaseAgent
from services.utils import format_agent_message
import json
import os

class NurseAgent(BaseAgent):
    def __init__(self, extractor: SlotExtractor, a2a_client=None):
        super().__init__("nurse")
        self.extractor = extractor
        self.a2a = a2a_client
        # Directory to persist slot states per session/thread
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sessions_dir = os.path.normpath(os.path.join(base_dir, "..", "data", "sessions"))
        os.makedirs(self.sessions_dir, exist_ok=True)

    def _slot_file_path(self, thread_id: str) -> str:
        safe_thread = str(thread_id).replace("/", "_").replace("\\", "_")
        return os.path.join(self.sessions_dir, f"{safe_thread}_slots.json")

    async def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        state: {'thread_id','query','messages','slots' (optional)}
        Returns a dict with:
          - type: "followup" or "complete"
          - question: the next question (for followup)
          - slots: the updated slots dict
          - text: textual representation (same as question)
        """
        text = state.get("query", "")
        thread_id = state.get("thread_id", "default")

        # Initialize full slot schema and preserve prior values across turns
        # Merge any state-provided prior with what's on disk for this thread
        prior_state = state.get("slots") or {}
        prior_file = {}
        slot_path = self._slot_file_path(thread_id)
        try:
            if os.path.exists(slot_path):
                with open(slot_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        prior_file = {k: data.get(k) for k in REQUIRED_SLOTS}
                        # also load negated_symptoms if present on disk
                        if isinstance(data.get("negated_symptoms"), list):
                            prior_file["negated_symptoms"] = data.get("negated_symptoms")
        except Exception:
            prior_file = {}
        prior = {k: (prior_state.get(k) if prior_state.get(k) is not None else prior_file.get(k)) for k in REQUIRED_SLOTS}
        collected = {k: prior.get(k, None) for k in REQUIRED_SLOTS}
        # ensure negated_symptoms exists and merge with any provided
        existing_neg = prior_state.get("negated_symptoms") if isinstance(prior_state.get("negated_symptoms"), list) else None
        file_neg = prior_file.get("negated_symptoms") if isinstance(prior_file.get("negated_symptoms"), list) else None
        merged_neg = []
        seen = set()
        for arr in (existing_neg or [] , file_neg or []):
            for item in arr:
                s = str(item).strip()
                low = s.lower()
                if s and low not in seen:
                    seen.add(low)
                    merged_neg.append(s)
        collected["negated_symptoms"] = merged_neg

        # Extract from latest user utterance (rule-based + LLM merge inside extractor)
        new = await self.extractor.extract_slots(text)

        # Merge strategy:
        # - If new value present, update
        # - For 'symptom', accumulate multiple distinct mentions (comma-separated)
        for k, v in (new or {}).items():
            if not v:
                continue
            if k == "symptom":
                prev = collected.get(k)
                if not prev:
                    collected[k] = v
                else:
                    # accumulate unique mentions
                    existing = [s.strip() for s in str(prev).split(",") if s.strip()]
                    incoming = [s.strip() for s in str(v).split(",") if s.strip()]
                    merged = []
                    seen = set()
                    for s in existing + incoming:
                        low = s.lower()
                        if low not in seen:
                            seen.add(low)
                            merged.append(s)
                    collected[k] = ", ".join(merged)
            elif k == "negated_symptoms":
                # merge as unique list (case-insensitive)
                prev_list = collected.get("negated_symptoms") or []
                merged = []
                seen = set()
                for s in list(prev_list) + list(v if isinstance(v, list) else []):
                    s_str = str(s).strip()
                    if not s_str:
                        continue
                    low = s_str.lower()
                    if low not in seen:
                        seen.add(low)
                        merged.append(s_str)
                collected["negated_symptoms"] = merged
            else:
                collected[k] = v

        # save back to state for other nodes
        state["slots"] = collected

        # Persist to JSON on every turn for durability and debugging
        try:
            to_save = {k: collected.get(k) for k in REQUIRED_SLOTS}
            to_save["negated_symptoms"] = collected.get("negated_symptoms", [])
            # Optional field aliases to match user-readable preferences
            # e.g., singular/plural variations kept in sync
            to_save_aliases = {
                "symptom": to_save.get("symptom"),
                "duration": to_save.get("duration"),
                "severity": to_save.get("severity"),
                "medical_history": to_save.get("medical_history"),
                "medication": to_save.get("medications"),
                "medications": to_save.get("medications"),
                "allergy": to_save.get("allergies"),
                "allergies": to_save.get("allergies"),
                "negated_symptoms": to_save.get("negated_symptoms", []),
            }
            with open(slot_path, "w", encoding="utf-8") as f:
                json.dump(to_save_aliases, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Non-fatal: continue even if persisting fails
            print(f"[NurseAgent] Failed to persist slot state to {slot_path}: {e}")

        # print/log current slot state for debugging
        pretty = json.dumps(collected, indent=2, ensure_ascii=False)
        print(f"\n--- Nurse Agent Slot State ---\n{pretty}\n-----------------------------\n")

        self.remember("user", text)

        # find missing slots (treat empty strings as missing)
        missing = [k for k in REQUIRED_SLOTS if not collected.get(k)]

        # Minimal sufficiency rule: if symptom and duration are present, we can complete
        has_minimal = bool(collected.get("symptom")) and bool(collected.get("duration"))
        if not has_minimal:
            if missing:
                # deterministically pick one question to ask next
                q = await self.extractor.next_question(collected, context=text)
                state["next_question"] = q
                # optionally notify research etc via A2A with structured slots
                if self.a2a:
                    self.a2a.send("nurse", "research", "triage_hint", thread_id, {"slots": collected})
                # return followup with updated full slots dict (not a list)
                return {"type": "followup", "question": q, "slots": collected, "text": q}
        else:
            # Auto-fill defaults for any remaining missing slots
            DEFAULTS = {
                "severity": "unspecified",
                "medical_history": "none reported",
                "medications": "none reported",
                "allergies": "none reported",
            }
            for k in missing:
                if k in DEFAULTS:
                    collected[k] = DEFAULTS[k]
                else:
                    collected[k] = "not provided"

        # minimal info present (symptom & duration) OR all slots present -> complete
        return {"type": "complete", "slots": collected, "text": "Collected sufficient information."}