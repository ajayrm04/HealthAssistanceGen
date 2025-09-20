# services/slot_extractor.py
import yaml, json, re
from typing import Dict, Any, Optional, List
from services.llm_adapter import LLMAdapter
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..","config", "config.yaml")
CFG = yaml.safe_load(open(config_path,"r",encoding="utf-8"))
REQUIRED_SLOTS = CFG.get("slots", {}).get("required", ["symptom","duration","severity","medical_history","medications","allergies"])

# ---------------- Rule-based helpers (fast, robust) ----------------



class SlotExtractor:
    def __init__(self, model_key: Optional[str] = None):
        model_name = None
        if model_key:
            model_name = CFG["llm"].get(model_key)
        self.llm = LLMAdapter(
            model_name=model_name,
            temperature=CFG["llm"].get("temperature", 0.7),
        )

    async def extract_slots(self, text: str) -> Dict[str, Any]:
        """
        LLM-only extraction: return a strict JSON object with REQUIRED_SLOTS plus
        an extra key "negated_symptoms" (array of strings). The "symptom" field
        MUST NOT include any negated items. If parsing fails, return REQUIRED_SLOTS
        as null and negated_symptoms as [].
        """
        keys_with_neg = REQUIRED_SLOTS + ["negated_symptoms"]
        system = (
            "You are a strict JSON slot extractor. Return ONLY a valid JSON object with exactly "
            f"these keys: {keys_with_neg}. Rules:\n"
            "- Values for required slots should be strings or null if unknown.\n"
            "- negated_symptoms MUST be an array of strings.\n"
            "- Identify symptoms the user explicitly denies (e.g., 'no chest pain',\n"
            "  'I don't have cold', 'denies fever') and list them in negated_symptoms.\n"
            "- Ensure 'symptom' DOES NOT include any item that appears in negated_symptoms.\n"
            "No extra text."
        )
        user = (
            "Extract the following patient information from the text.\n" \
            f"Text: \"{text}\"\n" \
            "Example valid response: {\n"
            "  \"symptom\": \"cough\",\n"
            "  \"duration\": \"3 days\",\n"
            "  \"severity\": null,\n"
            "  \"medical_history\": null,\n"
            "  \"medications\": null,\n"
            "  \"allergies\": null,\n"
            "  \"negated_symptoms\": [\"chest pain\", \"fever\"]\n"
            "}"
        )

        try:
            resp = await self.llm.simple(system, user)
            text_resp = str(resp).strip()
            # Best-effort JSON extraction if model adds surrounding text
            start = text_resp.find("{")
            end = text_resp.rfind("}")
            if start != -1 and end != -1 and end > start:
                text_resp = text_resp[start:end+1]
            data = json.loads(text_resp)
            out: Dict[str, Any] = {k: data.get(k, None) for k in REQUIRED_SLOTS}
            ns = data.get("negated_symptoms")
            if isinstance(ns, list):
                out["negated_symptoms"] = [str(x).strip() for x in ns if str(x).strip()]
            else:
                out["negated_symptoms"] = []
            return out
        except Exception:
            # As a last resort, try eval in a constrained way
            try:
                data = eval(text_resp)
                if isinstance(data, dict):
                    out: Dict[str, Any] = {k: data.get(k, None) for k in REQUIRED_SLOTS}
                    ns = data.get("negated_symptoms")
                    if isinstance(ns, list):
                        out["negated_symptoms"] = [str(x).strip() for x in ns if str(x).strip()]
                    else:
                        out["negated_symptoms"] = []
                    return out
            except Exception:
                pass
        out: Dict[str, Any] = {k: None for k in REQUIRED_SLOTS}
        out["negated_symptoms"] = []
        return out




## NOT BEING USED THO
    async def next_question(self, collected: Dict[str, Any], context: Optional[str] = "") -> str:
        """
        Use LLM to generate the next best question in a natural conversational style.
        Handles multiple symptoms if present.
        """
        # Determine missing slots
        missing = [k for k, v in collected.items() if not v]
        if not missing:
            return None

        # Deterministic fallback questions for common slots
        FALLBACK_QUESTIONS = {
            "symptom": "Could you describe your main symptom(s)?",
            "duration": "How long have you been experiencing this?",
            "severity": "How severe is it (mild, moderate, severe, or 1–10 rating)?",
            "medical_history": "Do you have any relevant medical history (e.g., diabetes, hypertension)?",
            "medications": "Are you currently taking any medications? If yes, which ones?",
            "allergies": "Do you have any allergies to medications or foods?",
        }

        next_slot = missing[0]

        # If symptom slot is missing and multiple symptoms are already collected, summarize them
        current_symptoms = collected.get("symptom")
        if next_slot == "symptom" and current_symptoms:
            if isinstance(current_symptoms, str) and "," in current_symptoms:
                current_symptoms_list = [s.strip() for s in current_symptoms.split(",") if s.strip()]
                context += f"\nCurrently identified symptoms: {', '.join(current_symptoms_list)}"

        # Prepare LLM prompt
        system_prompt = (
            "You are a friendly and empathetic medical assistant. "
            "Your task is to ask the next single question needed to complete a patient's triage. "
            "If multiple symptoms are mentioned, acknowledge them naturally. "
            "Keep the question concise, polite, and clear."
        )
        user_prompt = (
            f"Collected so far: {json.dumps(collected)}\n"
            f"Missing slot: {next_slot}\n"
            f"Context: {context}\n"
            f"Return ONLY the next question text."
        )

        # Call LLM
        try:
            print("----NURSE----NURSE----NURSE----NURSE----NURSE----NURSE----")
            resp = await self.llm.simple(system_prompt, user_prompt)

            question = str(resp).strip()
            print(question)
            if not question:
                # fallback deterministic question
                question = FALLBACK_QUESTIONS.get(next_slot, "Could you provide more details?")
        except Exception:
            question = FALLBACK_QUESTIONS.get(next_slot, "Could you provide more details?")

        return question

