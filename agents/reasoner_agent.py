# healthcare_agents/agents/reasoner_agent.py

from typing import List, Dict, Any
from services.llm_adapter import LLMAdapter
from services.slot_extractor import REQUIRED_SLOTS
from services.kg_service import KGService
import json

class ReasonerAgent:
    """
    Fuses multi-agent outputs (Doctor, Research, Nurse).
    Performs reasoning, resolves conflicts, before ComplianceAgent sees them.
    """
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def reason(self, messages: List, slots: Dict[str, Any]):
        """
        Use KG-derived disease/symptom relations to ask a discriminative follow-up
        question when slots are not yet complete. If all REQUIRED_SLOTS are filled,
        return an empty string to allow the doctor/compliance flow to proceed without
        additional probing from the reasoner.
        """
        try:
            # Normalize inputs and collect KG payloads up front
            slots = slots or {}
            text_blobs = [getattr(m, "content", str(m)) for m in (messages or [])]
            kg_payloads = []
            for t in text_blobs:
                t_str = str(t)
                if t_str.startswith("[nurse_kg]") or t_str.startswith("[doctor_kg]"):
                    try:
                        json_part = t_str.split("]", 1)[1].strip()
                        data = json.loads(json_part)
                        if isinstance(data, dict) and data.get("triples"):
                            kg_payloads.append(data)
                    except Exception:
                        continue

            # Build disease->symptoms map from kg_payloads if present
            disease_to_symptoms: Dict[str, List[str]] = {}
            if kg_payloads:
                all_triples = []
                for payload in kg_payloads:
                    triples = payload.get("triples") or []
                    if isinstance(triples, list):
                        for item in triples:
                            if isinstance(item, (list, tuple)) and len(item) == 3:
                                all_triples.append((str(item[0]), str(item[1]), str(item[2])))
                kg = KGService()
                try:
                    disease_to_symptoms = kg.get_all_symptoms_for_diseases_from_triples(all_triples)
                finally:
                    kg.close()

            # 1) If all required slots are filled, produce probable diseases with symptoms
            all_filled = all(bool(slots.get(k)) for k in REQUIRED_SLOTS)
            if all_filled:
                # Prefer KG when available; otherwise use LLM knowledge
                if disease_to_symptoms:
                    system = (
                        "You are a clinical reasoning assistant. Use the provided knowledge graph "
                        "disease-to-symptoms mapping and the completed patient slots to infer the "
                        "most probable diseases. Return a concise ranked list where each item contains: "
                        "Disease name and a short list of hallmark symptoms from the KG that match."
                    )
                    user = json.dumps({
                        "completed_slots": slots,
                        "kg": disease_to_symptoms,
                    }, ensure_ascii=False)
                    return await self.llm.simple(system, f"Context:\n{user}\nReturn the ranked list:")
                else:
                    print("HIIIII")
                    system = (
                        "Just tell the user that the data recieved hasnt matched any disease in a neat way"
                    )
                    return await self.llm.simple(system, json.dumps(slots, ensure_ascii=False))

            # 2) If slots are incomplete and we have KG data, ask a discriminative question
            #    and additional questions to fill any missing REQUIRED_SLOTS
            # try:
            #     print("[KG payloads]", json.dumps(kg_payloads, ensure_ascii=False, indent=2))
            # except Exception:
            #     print("[KG payloads]", kg_payloads)
            if disease_to_symptoms:
                missing_slots = [k for k in REQUIRED_SLOTS if not slots.get(k)]
                system_prompt = (
                    "You are a clinical reasoning assistant. Using the provided knowledge graph "
                    "disease-to-symptoms mapping and the user's currently reported symptoms, do two things:\n"
                    "1) Generate ONE short, specific discriminative question to decide between the top candidate diseases.\n"
                    "2) Generate concise slot-filling questions for EACH missing slot listed.\n"
                    "Formatting rules:\n"
                    "- Output multiple questions, each on its own line\n"
                    "- First line MUST be the discriminative question\n"
                    "- Second person (e.g., 'Do you have…?')\n"
                    "- Plain English, clinically accurate\n"
                    "- No preamble or explanations\n"
                )
                try:
                    print("[Disease→Symptoms]", json.dumps(disease_to_symptoms, ensure_ascii=False, indent=2))
                    print("Number of diseases:",len(disease_to_symptoms))
                except Exception:
                    print("[Disease→Symptoms]", disease_to_symptoms)
                user_payload = {
                    "reported_symptoms": slots.get("symptom"),
                    "kg": disease_to_symptoms,
                    "missing_slots": missing_slots,
                }
                user_prompt = (
                    "Data:\n" + json.dumps(user_payload, ensure_ascii=False) + "\n\n"
                    "Return ONLY the questions, one per line."
                )
                question = await self.llm.simple(system_prompt, user_prompt)
                return str(question).strip()

            # 3) Fallback: minimal pass-through reasoning request
            prompt = (
                "You are the medical reasoning agent. Review the multi-agent outputs and propose "
                "ONE concise follow-up question that helps gather the next most useful detail. "
                "Return ONLY the question."
            )
            return await self.llm.simple("", f"Messages:\n{[str(b) for b in text_blobs]}\nQuestion:")
        except Exception:
            return ""


# healthcare_agents/agents/reasoner_agent.py

from typing import List
from services.llm_adapter import LLMAdapter

# class ReasonerAgent:
#     """
#     Fuses multi-agent outputs (Doctor, Research, Nurse).
#     Performs reasoning, resolves conflicts, before ComplianceAgent sees them.
#     """
#     def __init__(self, llm: LLMAdapter):
#         self.llm = llm

#     async def reason(self, messages: List):
#         prompt = f"""
#         You are the medical reasoning agent.
#         Review the following multi-agent outputs:
#         {messages}
        
#         - Resolve conflicts if present
#         - Add differential reasoning
#         - Prepare a single coherent draft answer
#         - Do NOT give final advice; that goes through compliance

#         Return your reasoning draft as plain text.
#         """
#         return await self.llm.simple("", prompt)
