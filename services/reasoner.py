# services/reasoner.py
from typing import Dict, Any, Optional
import yaml
from services.llm_adapter import LLMAdapter
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..","config", "config.yaml")
CFG = yaml.safe_load(open(config_path,"r",encoding="utf-8"))

class MCPReasoner:
    def __init__(self, llm: LLMAdapter = None, model_key: str = None):
        # Accept either an LLMAdapter instance or create one from model_key
        if llm is not None:
            self.llm = llm
        else:
            model_name = None
            if model_key:
                model_name = CFG["llm"].get(model_key)
            self.llm = LLMAdapter(model_name=model_name, temperature=CFG["llm"].get("temperature"), max_tokens=CFG["llm"].get("max_tokens"))

    async def reason(self, mcp_payload: Dict[str,Any], patient_meta: Optional[Dict[str,Any]] = None) -> str:
        evs = mcp_payload.get("evidence", [])
        evidence_text = "\n".join([f"[{ev['id']} | {ev['source']}] {ev['content']}" for ev in evs]) or "<none>"
        patient_block = "\n".join([f"{k}: {v}" for k,v in (patient_meta or {}).items()]) or "<none>"

        system = ("You are a cautious medical assistant. Use ONLY the provided evidence and patient metadata. "
                  "Cite evidence IDs for claims. If insufficient, ask for more information. Provide: short summary; possible conditions (with evidence refs); recommended next steps; confidence; disclaimer.")
        user = f"Question: {mcp_payload['question']}\nPatient:\n{patient_block}\nEVIDENCE:\n{evidence_text}\nAnswer succinctly and cite evidence IDs."

        return await self.llm.simple(system, user)

    async def differential(self, mcp_payload: Dict[str,Any], patient_meta: Optional[Dict[str,Any]] = None) -> str:
        evs = mcp_payload.get("evidence", [])
        evidence_text = "\n".join([f"[{ev['id']} | {ev['type']}] {ev['content']}" for ev in evs]) or "<none>"
        kg_only = [ev for ev in evs if ev.get("type") == "kg"]
        kg_text = "\n".join([f"[{ev['id']}] {ev['content']}" for ev in kg_only]) or "<none>"
        patient_block = "\n".join([f"{k}: {v}" for k,v in (patient_meta or {}).items()]) or "<none>"

        system = (
            "You are a clinical reasoning assistant. Based on the knowledge-graph triples and other evidence, "
            "produce a concise differential diagnosis list. For each candidate condition: give a one-line rationale referencing KG triples by ID if applicable. "
            "Return: 1) top 3-5 differentials ranked; 2) immediate red-flags; 3) suggested tests."
        )
        user = (
            f"Question: {mcp_payload['question']}\n"
            f"Patient:\n{patient_block}\n"
            f"KG triples:\n{kg_text}\n"
            f"Other evidence:\n{evidence_text}\n"
            f"Respond succinctly with bullet points and cite KG IDs."
        )
        return await self.llm.simple(system, user)

    async def classify_route(self, query: str) -> str:
        system = "You are a router. Choose one token only: kg-only|vdb-only|parallel"
        user = f"Query: {query}"
        out = await self.llm.simple(system, user)
        out = str(out).strip().lower()
        return out if out in {"kg-only","vdb-only","parallel"} else "parallel"
