# agents/research_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent

class ResearchAgent(BaseAgent):
    def __init__(self, vdb_service):
        super().__init__("research")
        self.vdb = vdb_service

    def handle_a2a(self, envelope: Dict[str,Any]) -> Dict[str,Any]:
        # envelope.payload may contain 'slots' or 'query'
        p = envelope.get("payload",{})
        q = p.get("query") or " ".join([str(v) for v in p.get("slots",{}).values() if v])
        notes = []
        if not q:
            return {"status":"ok","notes":[]}
        # quick evidence from VDB using query text
        hits = self.vdb.query(q, top_k=3)
        for t,s in hits:
            notes.append(t)
        return {"status":"ok","notes": notes}

    async def handle(self, state: Dict[str,Any]) -> str:
        # return research notes if invoked directly
        q = state.get("query","")
        hits = self.vdb.query(q, top_k=3)
        notes = [t for t,s in hits]
        return "\n".join(notes) if notes else "No research notes found."
