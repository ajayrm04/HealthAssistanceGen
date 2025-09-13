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
COMMON_SYMPTOMS: List[str] = [
    "headache","fever","cough","chest pain","shortness of breath","sob","nausea",
    "vomiting","diarrhea","dizziness","fatigue","sore throat","back pain","rash",
    "abdominal pain","stomach ache","runny nose","congestion","chills","body ache"
]
SYMPTOM_PATTERNS = [
    r"\b(i have|i'm having|i am having|i feel|i'm feeling|i am feeling|experiencing|with)\s+([a-z\s-]+?)\b(?:for|since|and|but|\.|,|$)",
]

DURATION_PATTERNS = [
    r"\bfor\s+(?:about\s+)?(\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?))\b",
    r"\b(\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?))\b",
    r"\bsince\s+(yesterday|today|last night|last week|last month)\b",
    r"\bfor\s+(a\s+few|a\s+couple\s+of)\s+(days?|weeks?|months?)\b",
]

SEVERITY_PATTERNS = [
    r"\b(mild|moderate|severe|worst)\b",
    r"\b(\d+)\s*/\s*10\b",
    r"\b(severity|pain)\s*(\d+)\b",
]

ALLERGY_PATTERN = r"\ballergic\s+to\s+([a-zA-Z0-9 ,-/]+)"
MEDICATION_PATTERNS = [
    r"\b(taking|on|started|start|took)\s+([a-zA-Z0-9 ,-/]+)\b",
]

COMMON_MEDICATIONS: List[str] = [
    "aspirin","ibuprofen","paracetamol","acetaminophen","metformin","amoxicillin",
    "atorvastatin","lisinopril","omeprazole","insulin","albuterol","prednisone"
]

COMMON_CONDITIONS: List[str] = [
    "hypertension","high blood pressure","diabetes","asthma","copd","coronary artery disease",
    "heart failure","stroke","cancer","kidney disease","liver disease","thyroid disorder",
    "depression","anxiety","high cholesterol","hyperlipidemia"
]

def _match_first(patterns: List[str], text: str) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            # pick last group with content
            groups = [g for g in m.groups() if g]
            if groups:
                return groups[-1].strip()
            return m.group(0).strip()
    return None

def _extract_symptom(text: str) -> Optional[str]:
    # Try explicit patterns like "I have X" first
    m = _match_first(SYMPTOM_PATTERNS, text)
    if m:
        return m
    # Otherwise scan for common symptom keywords
    for sym in COMMON_SYMPTOMS:
        if re.search(rf"\b{re.escape(sym)}\b", text, flags=re.IGNORECASE):
            return sym
    return None

def _extract_duration(text: str) -> Optional[str]:
    m = _match_first(DURATION_PATTERNS, text)
    return m

def _extract_severity(text: str) -> Optional[str]:
    # e.g., "pain 7/10" or words
    m = _match_first(SEVERITY_PATTERNS, text)
    return m

def _extract_allergies(text: str) -> Optional[str]:
    m = re.search(ALLERGY_PATTERN, text, flags=re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip('.')
        # cleanup trailing noise
        val = re.sub(r"\b(and|but|for|since)\b.*$", "", val, flags=re.IGNORECASE).strip(', ').strip()
        return val or None
    return None

def _extract_medications(text: str) -> Optional[str]:
    # Look for explicit phrasing first
    for pat in MEDICATION_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            candidate = m.group(2).strip().strip('.')
            candidate = re.sub(r"\b(and|but|for|since)\b.*$", "", candidate, flags=re.IGNORECASE).strip(', ').strip()
            if candidate:
                return candidate
    # Otherwise, search for known meds mentioned in text and join
    hits = []
    for med in COMMON_MEDICATIONS:
        if re.search(rf"\b{re.escape(med)}\b", text, flags=re.IGNORECASE):
            hits.append(med)
    if hits:
        return ", ".join(sorted(set(hits)))
    return None

def _extract_med_history(text: str) -> Optional[str]:
    # Look for "history of X"
    m = re.search(r"history of ([a-zA-Z0-9 ,-/]+)", text, flags=re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip('.')
        val = re.sub(r"\b(and|but|for|since)\b.*$", "", val, flags=re.IGNORECASE).strip(', ').strip()
        return val or None
    # Otherwise scan for known conditions as list
    hits = []
    for cond in COMMON_CONDITIONS:
        if re.search(rf"\b{re.escape(cond)}\b", text, flags=re.IGNORECASE):
            hits.append(cond)
    if hits:
        return ", ".join(sorted(set(hits)))
    return None

def _rule_extract(text: str) -> Dict[str, Any]:
    return {
        "symptom": _extract_symptom(text),
        "duration": _extract_duration(text),
        "severity": _extract_severity(text),
        "medical_history": _extract_med_history(text),
        "medications": _extract_medications(text),
        "allergies": _extract_allergies(text),
    }


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
        LLM-only extraction: ask the model to return a strict JSON object with
        the required keys. If the model output cannot be parsed, return all keys with null.
        """
        system = (
            "You are a strict JSON slot extractor. Return ONLY a valid JSON object with exactly "
            f"these keys: {REQUIRED_SLOTS}. Values should be strings or null if unknown. No extra text."
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
            "  \"allergies\": null\n"
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
            return {k: data.get(k, None) for k in REQUIRED_SLOTS}
        except Exception:
            # As a last resort, try eval in a constrained way
            try:
                data = eval(text_resp)
                if isinstance(data, dict):
                    return {k: data.get(k, None) for k in REQUIRED_SLOTS}
            except Exception:
                pass
        return {k: None for k in REQUIRED_SLOTS}

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

