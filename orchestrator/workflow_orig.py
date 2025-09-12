# graph/workflow.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from typing import List, Annotated
from langgraph.pregel import StateGraph, add_messages
from langchain.schema import BaseMessage

class OrchestratorState:
    # Accumulate messages instead of overwriting
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str | None = None
class AppState(TypedDict, total=False):
    question: str
    route: str
    result: Any

class Orchestrator:
    def __init__(self, doctor, nurse, research, compliance, reasoner):
        self.doctor = doctor
        self.nurse = nurse
        self.research = research
        self.compliance = compliance
        self.reasoner = reasoner

    async def router(self, state: AppState) -> dict:
        q = state["question"]
        route = await self.reasoner.classify_route(q)
        return {"route": route}

    async def run_doctor(self, state: AppState) -> dict:
        out = await self.doctor.run(state["question"])
        return {"result": out}

    async def run_nurse(self, state: AppState) -> dict:
        out = await self.nurse.run(state["question"])
        return {"result": out}

    async def run_research(self, state: AppState) -> dict:
        out = await self.research.run(state["question"])
        return {"result": out}

    async def run_compliance(self, state: AppState) -> dict:
        agent_out = state.get("result", {})
        answer = agent_out.get("answer", "")
        comp = await self.compliance.run(answer)
        agent_out["compliance"] = comp
        return {"result": agent_out}

def build_workflow(orchestrator: Orchestrator):
    g = StateGraph(AppState)
    g.add_node("router", orchestrator.router)
    g.add_node("doctor", orchestrator.run_doctor)
    g.add_node("nurse", orchestrator.run_nurse)
    g.add_node("research", orchestrator.run_research)
    g.add_node("compliance", orchestrator.run_compliance)

    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s["route"], {"doctor":"doctor","nurse":"nurse","research":"research"})
    g.add_edge("doctor","compliance")
    g.add_edge("nurse","compliance")
    g.add_edge("research","compliance")
    g.add_edge("compliance", END)
    return g.compile()
