# healthcare_agents/agents/reasoner_agent.py

from typing import List, Dict, Any
from services.llm_adapter import LLMAdapter
from services.slot_extractor import REQUIRED_SLOTS
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

            # 1) If all required slots are filled, produce probable diseases with symptoms
            all_filled = all(bool(slots.get(k)) for k in REQUIRED_SLOTS)
            if all_filled:
                # Prefer KG when available; otherwise use LLM knowledge
                if kg_payloads:
                    system = (
                        "You are a clinical reasoning assistant. Use the provided knowledge graph "
                        "triples (disease–symptom edges) and the completed patient slots to infer the "
                        "most probable diseases. Return a concise ranked list where each item contains: "
                        "Disease name and a short list of hallmark symptoms from the KG that match."
                    )
                    user = json.dumps({
                        "completed_slots": slots,
                        "kg": kg_payloads,
                    }, ensure_ascii=False)
                    return await self.llm.simple(system, f"Context:\n{user}\nReturn the ranked list:")
                else:
                    system = (
                        "You are a clinical reasoning assistant. Based on the completed triage slots, "
                        "list the most probable diseases and include their hallmark symptoms. Be concise."
                    )
                    return await self.llm.simple(system, json.dumps(slots, ensure_ascii=False))

            # 2) If slots are incomplete and we have KG data, ask one discriminative question
            if kg_payloads:
                system_prompt = (
                    "You are a clinical reasoning assistant. Using the provided knowledge graph "
                    "triples (disease–symptom relations) and the user's currently reported symptoms, "
                    "generate ONE short, specific discriminative question that helps decide between the "
                    "most likely diseases. Choose a symptom or feature that is present in one strong candidate "
                    "disease but absent in competing candidates. The question must be:\n"
                    "- One sentence\n"
                    "- Second person (e.g., 'Do you have…?')\n"
                    "- Plain English, clinically accurate\n"
                    "- No preamble, no explanation, no lists, no extra text\n"
                )
                user_payload = {
                    "reported_symptoms": slots.get("symptom"),
                    "kg": kg_payloads,
                }
                user_prompt = (
                    "Data:\n" + json.dumps(user_payload, ensure_ascii=False) + "\n\n"
                    "Return ONLY the question text."
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
