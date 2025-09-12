# healthcare_agents/orchestrator/orchestrator.py
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from orchestrator.state_schema import OrchestratorState
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Fallback if langgraph isn't available
    class MemorySaver:
        def __init__(self):
            self.memory = {}
            
        def save(self, key, value):
            self.memory[key] = value
            
        def load(self, key):
            return self.memory.get(key)
            
from typing import Dict, Any, List
from services.slot_extractor import SlotExtractor
from .workflow import build_workflow

# Agents
from agents.nurse_agent import NurseAgent
from agents.doctor_agent import DoctorAgent
from agents.research_agent import ResearchAgent
from agents.compliance_agent import ComplianceAgent
from agents.router_agent import RouterAgent
from agents.reasoner_agent import ReasonerAgent

# Services
from services.llm_adapter import LLMAdapter
from services.mcp import MCPAssembler
from services.reasoner import MCPReasoner
from services.vdb_service import VDBService
from services.kg_service import KGService


def normalize_messages(messages) -> List[BaseMessage]:
    """
    Normalize raw history into LangChain message objects.
    Supports dicts, tuples, and already-typed messages.
    """
    normalized: List[BaseMessage] = []
    for m in messages or []:
        if isinstance(m, BaseMessage):
            normalized.append(m)

        elif isinstance(m, tuple):
            role, content = m
            if role in ["user", "human"]:
                normalized.append(HumanMessage(content=content))
            elif role in ["assistant", "ai", "doctor", "nurse", "research", "reasoner"]:
                normalized.append(AIMessage(content=f"[{role}] {content}"))
            elif role == "system" or role == "compliance":
                normalized.append(SystemMessage(content=f"[{role}] {content}"))
            else:
                normalized.append(AIMessage(content=f"[unknown:{role}] {content}"))

        elif isinstance(m, dict):
            role = m.get("role", "unknown")
            content = m.get("content") or m.get("text", "")
            if role in ["user", "human"]:
                normalized.append(HumanMessage(content=content))
            elif role in ["assistant", "ai", "doctor", "nurse", "research", "reasoner"]:
                normalized.append(AIMessage(content=f"[{role}] {content}"))
            elif role == "system" or role == "compliance":
                normalized.append(SystemMessage(content=f"[{role}] {content}"))
            else:
                normalized.append(AIMessage(content=f"[unknown:{role}] {content}"))

        else:
            normalized.append(AIMessage(content=str(m)))
    return normalized


class Orchestrator:
    """
    Orchestrates multi-agent healthcare workflow:
    User → Router → {Nurse, Doctor, Research} → Reasoner → Compliance → User
    """

    def __init__(self):
        # Init shared services
        self.llm = LLMAdapter(model_name="gpt-4o")
        self.kg = KGService()
        self.vdb = VDBService()
        assembler = MCPAssembler()
        reasoner = MCPReasoner(self.llm)

        # Agents
        self.router = RouterAgent(self.llm)
        self.nurse = NurseAgent(SlotExtractor())
        self.doctor = DoctorAgent(self.kg, self.vdb, assembler=assembler, reasoner=reasoner)
        self.research = ResearchAgent(self.vdb)
        self.reasoner = ReasonerAgent(self.llm)
        self.compliance = ComplianceAgent()

        # Persist triage slots per thread across turns
        self.thread_slots: Dict[str, Any] = {}

        # Graph + checkpointing
        self.checkpointer = MemorySaver()
        self.workflow = build_workflow(
            router_node=self.router_node,
            nurse_node=self.nurse_node,
            doctor_node=self.doctor_node,
            research_node=self.research_node,
            reasoner_node=self.reasoner_node,
            compliance_node=self.compliance_node,
        )
        self.graph = self.workflow.compile(checkpointer=self.checkpointer)

    # ------------------ Node wrappers ------------------

    async def router_node(self, state: OrchestratorState) -> Dict[str, Any]:
        routes = await self.router.route(state.user_query, state.messages)
        return {"routes": routes, "messages": state.messages, "slots": state.slots}

    async def nurse_node(self, state: OrchestratorState) -> Dict[str, Any]:
        resp = await self.nurse.handle({
            "thread_id": state.thread_id,
            "query": state.user_query,
            "messages": state.messages,
            "slots": state.slots
        })
        updated = state.messages + [AIMessage(content=f"[nurse] {resp.get('text', resp)}")]
        return {
            "messages": normalize_messages(updated),
            "slots": resp.get("slots", state.slots)
        }

    async def doctor_node(self, state: OrchestratorState) -> Dict[str, Any]:
        resp = await self.doctor.handle({
            "thread_id": state.thread_id,
            "query": state.user_query,
            "messages": state.messages,
            "slots": state.slots
        })
        # Attach a structured KG payload message for the reasoner to consume
        try:
            import json
            kg_payload = json.dumps({"symptom": state.slots.get("symptom"), "triples": resp.get("kg_triples", [])}, ensure_ascii=False)
            updated = state.messages + [AIMessage(content=f"[doctor] {resp['answer']}"), AIMessage(content=f"[doctor_kg] {kg_payload}")]
        except Exception:
            updated = state.messages + [AIMessage(content=f"[doctor] {resp['answer']}")]
        return {"messages": normalize_messages(updated)}

    async def research_node(self, state: OrchestratorState) -> Dict[str, Any]:
        resp = await self.research.handle({
            "thread_id": state.thread_id,
            "query": state.user_query,
            "messages": state.messages
        })
        updated = state.messages + [AIMessage(content=f"[research] {resp}")]
        return {"messages": normalize_messages(updated)}

    async def reasoner_node(self, state: OrchestratorState) -> Dict[str, Any]:
        fused = await self.reasoner.reason(state.messages)
        updated = state.messages + [AIMessage(content=f"[reasoner] {fused}")]
        return {"messages": normalize_messages(updated)}

    async def compliance_node(self, state: OrchestratorState) -> Dict[str, Any]:
        draft = state.messages[-1].content if state.messages else ""
        result = await self.compliance.handle({
            "draft": draft,
            "thread_id": state.thread_id,
            "query": state.user_query,   # ✅ use user_query, not slots.get()
        })
        if result["type"] == "approved":
            final = result["final"]
        else:  # escalated
            final = f"⚠️ Escalated: {', '.join(result.get('issues', []))}"

        updated = state.messages + [("compliance", final)]
        return {"messages": normalize_messages(updated), "final_response": final}



    # ------------------ Public API ------------------

    async def run_turn(self, thread_id: str, user_query: str, prior_messages=None) -> dict:
        """
        Run one turn of conversation through the orchestrator graph.
        Always return a normalized dict with at least {"text": ...}.
        """
        prior_messages = normalize_messages(
            prior_messages or [HumanMessage(content=user_query)]
        )
        init_state = OrchestratorState(
            thread_id=thread_id,
            user_query=user_query,
            messages=prior_messages,
            slots=self.thread_slots.get(thread_id, {})
        )
        res = await self.graph.ainvoke(
            init_state, config={"configurable": {"thread_id": thread_id}}
        )

        # Normalize response
        final = res.get("final_response")
        # Persist latest slots for this thread
        if isinstance(res, dict) and "slots" in res:
            self.thread_slots[thread_id] = res.get("slots") or self.thread_slots.get(thread_id, {})
        return {
            "text": str(final),
            "raw": res  # keep the full thing in case frontend needs more info
        }

