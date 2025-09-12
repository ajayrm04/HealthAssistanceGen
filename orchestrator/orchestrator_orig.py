from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import Dict, Any
import yaml, os
from services.a2a import A2AClient
from services.slot_extractor import SlotExtractor
from services.mcp import MCPAssembler
from services.reasoner import MCPReasoner
import os
import asyncio

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..", "config", "config.yaml")
CFG = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

class Orchestrator:
    def __init__(self, kg_service, vdb_service):
        self.a2a = A2AClient(enabled=CFG["a2a"]["enabled"], transport=CFG["a2a"]["transport"], http_endpoint=CFG["a2a"]["http_endpoint"])
        self.slot_extractor = SlotExtractor(model_key=CFG["llm"]["nurse_model_key"])
        self.assembler = MCPAssembler(max_tokens=CFG["mcp"]["max_tokens"], max_items=CFG["mcp"]["max_items"], prefer_kg_boost=CFG["mcp"]["prefer_kg_score_boost"])
        self.reasoner = MCPReasoner(model_key=CFG["llm"]["reasoner_model_key"])
        
        from agents.nurse_agent import NurseAgent
        from agents.doctor_agent import DoctorAgent
        from agents.research_agent import ResearchAgent
        from agents.compliance_agent import ComplianceAgent
        
        self.nurse = NurseAgent(self.slot_extractor, a2a_client=self.a2a)
        self.research = ResearchAgent(vdb_service)
        self.kg = kg_service
        self.vdb = vdb_service
        self.doctor = DoctorAgent(self.kg, self.vdb, self.assembler, self.reasoner, a2a_client=self.a2a)
        self.compliance = ComplianceAgent()

        if self.a2a.transport == "local" and self.a2a.enabled:
            self.a2a.register_local("nurse", lambda env: self.nurse_a2a_handler(env))
            self.a2a.register_local("research", lambda env: self.research.handle_a2a(env))
            self.a2a.register_local("compliance", lambda env: self.compliance.handle_a2a(env) if hasattr(self.compliance, "handle_a2a") else {"status": "ok"})

        self.graph = self._build_graph()

    def nurse_a2a_handler(self, env):
        payload = env.get("payload", {})
        text = payload.get("text", "")
        out = asyncio.get_event_loop().run_until_complete(self.slot_extractor.extract_slots(text))
        return {"status": "ok", "slots": out}

    async def _router(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mode = await self.reasoner.classify_route(state.get("query", ""))
        state["route"] = mode
        return state

    async def _retrieve_kg(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("retrieval", {})["kg_triples"] = self.kg.retrieve_triples(state.get("query", ""), limit=CFG["faiss"]["top_k"])
        return state

    async def _retrieve_vdb(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("retrieval", {})["vdb_chunks"] = self.vdb.query(state.get("query", ""), top_k=CFG["faiss"]["top_k"])
        return state

    async def _assess(self, state: Dict[str, Any]) -> Dict[str, Any]:
        kg = state.get("retrieval", {}).get("kg_triples", [])
        vdb = state.get("retrieval", {}).get("vdb_chunks", [])
        vdb_evs = self.assembler.from_vdb(vdb)
        kg_evs = self.assembler.from_kg(kg)
        combined = self.assembler.dedupe_and_rank(vdb_evs + kg_evs)
        state["mcp"] = self.assembler.assemble_context(combined, state.get("query", ""))
        state["info_sufficient"] = len(state["mcp"]["evidence"]) >= 2
        return state

    async def _nurse(self, state: Dict[str, Any]) -> Dict[str, Any]:
        res = await self.nurse.handle(state)
        state.update(res)
        return state

    async def _doctor(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = await self.doctor.handle(state)
        state.update(out)
        return state
    async def doctor_node(state: OrchestratorState):        
        answer = await doctor_agent.run(state.user_query)
        return {"messages": [answer]}   # ✅ goes into add_messages, not __root__


    async def _compliance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = await self.compliance.handle({"thread_id": state.get("thread_id"), "query": state.get("query"), "draft": state.get("answer") or state.get("final_answer") or state.get("mcp")})
        state["compliance"] = out
        return state

    def _build_graph(self):
        g = StateGraph(dict)
        g.add_node("router", self._router)
        g.add_node("retrieve_kg", self._retrieve_kg)
        g.add_node("retrieve_vdb", self._retrieve_vdb)
        g.add_node("assess", self._assess)
        g.add_node("nurse", self._nurse)
        g.add_node("doctor", self._doctor)
        g.add_node("compliance", self._compliance)

        g.set_entry_point("router")

        def route_decider(state):
            mode = state.get("route", "parallel")
            if mode == "kg-only": return ["retrieve_kg"]
            if mode == "vdb-only": return ["retrieve_vdb"]
            return ["retrieve_kg", "retrieve_vdb"]
        
        g.add_conditional_edges("router", route_decider)
        g.add_edge("retrieve_kg", "assess")
        g.add_edge("retrieve_vdb", "assess")
        g.add_edge("assess", "nurse")
        g.add_edge("nurse", "doctor")
        g.add_edge("doctor", "compliance")
        g.add_edge("compliance", END)

        return g.compile()

    async def run_turn(self, thread_id: str, user_text: str, prior_messages=None):
        db = CFG["app"]["conversation_db"]
        os.makedirs(os.path.dirname(db) or ".", exist_ok=True)
        checkpointer = AsyncSqliteSaver(db)
        
        graph_with_checkpoint = self.graph.with_config(checkpointer=checkpointer, configurable={"thread_id": thread_id})
        
        state = {"messages": prior_messages or [], "query": user_text}
        
        res = await graph_with_checkpoint.ainvoke(state)
        
        if res.get("next_question"):
            return {"type": "followup", "text": res["next_question"], "mcp": res.get("mcp")}
        if res.get("answer"):
            return {"type": "answer", "text": res.get("answer"), "mcp": res.get("mcp"), "compliance": res.get("compliance")}
        
        return {"type": "answer", "text": res.get("final_answer") or "No answer", "mcp": res.get("mcp"), "compliance": res.get("compliance")}