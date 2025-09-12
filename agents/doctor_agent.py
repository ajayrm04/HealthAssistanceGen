# agents/doctor_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent
from services.mcp import MCPAssembler
from services.reasoner import MCPReasoner
from services.utils import format_agent_message
from services.kg_service import KGService
from services.vdb_service import VDBService

vdb_service = VDBService()
kg_service = KGService()

class DoctorAgent(BaseAgent):
    def __init__(self, kg_service, vdb_service, assembler: MCPAssembler, reasoner: MCPReasoner, a2a_client=None):
        super().__init__("doctor")
        self.kg = kg_service
        self.vdb = vdb_service
        self.assembler = assembler
        self.reasoner = reasoner
        self.a2a = a2a_client

    async def handle(self, state: Dict[str,Any]) -> Dict[str,Any]:
        # state contains slots collected by nurse
        slots = state.get("slots", {})
        query = state.get("query", "Patient consultation")
        self.remember("system", f"Handling case for slots: {slots}")
        
        # Build a search query from slots (prefer symptom only as requested)
        search_q = (slots.get("symptom") or "").strip()
        if not search_q:
            # Fallback to concatenating all values if symptom missing
            search_q = " ".join([str(v) for v in slots.values() if v])
        print(slots.values())
        print("Search", search_q)
        print("Query",query)
        # Retrieve in parallel
        kg_triples = self.kg.retrieve_triples(search_q, limit=20)
        vdb_hits = self.vdb.query(search_q, top_k=8)
        vdb_evs = self.assembler.from_vdb(vdb_hits)
        kg_evs = self.assembler.from_kg(kg_triples)
        combined = self.assembler.dedupe_and_rank(vdb_evs + kg_evs)
        mcp = self.assembler.assemble_context(combined, question=search_q)
        # Compose KG-only output (no external LLM knowledge)
        kg_only_lines = [f"({s}) -[{p}]-> ({o})" for (s,p,o) in kg_triples] or ["<no KG triples found>"]
        kg_only_text = "\n".join(kg_only_lines)
        answer = f"KG findings for symptom '{search_q}':\n{kg_only_text}"
        differential = "Derived strictly from KG relations above."
        msg = format_agent_message("doctor", answer)
        # Ask research for evidence notes via A2A if configured
        research_notes = []
        if self.a2a:
            r = self.a2a.send("doctor","research","evidence_hints", state.get("thread_id",""), {"query":search_q})
            research_notes = r.get("notes", [])
        return {"type":"answer","mcp":mcp,"answer":answer,"differential": differential,"research_notes":research_notes,"kg_triples": kg_triples, "messages": [msg]}