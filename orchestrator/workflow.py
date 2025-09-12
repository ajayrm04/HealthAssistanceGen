from typing import List, Annotated, Optional
try:
    from langgraph.graph import StateGraph
    from langgraph.graph.message import add_messages
except ImportError:
    # Fallback if langgraph isn't available - define minimal StateGraph for testing
    class StateGraph:
        def __init__(self, state_type):
            self.state_type = state_type
            self.nodes = {}
            self.edges = []
            
        def add_node(self, name, func):
            self.nodes[name] = func
            
        def add_edge(self, from_node, to_node):
            self.edges.append((from_node, to_node))
            
        def add_conditional_edges(self, from_node, condition_func, condition_map):
            self.edges.append((from_node, condition_func, condition_map))
            
        def set_entry_point(self, node):
            self.entry_point = node
            
        def set_finish_point(self, node):
            self.finish_point = node
            
        def compile(self, checkpointer=None):
            return self
            
    def add_messages(messages, new_messages):
        return messages + new_messages
        
from langchain_core.messages import BaseMessage
from .state_schema import OrchestratorState

# State definition: accumulate all messages in `messages`


def build_workflow(router_node, nurse_node, doctor_node, research_node, reasoner_node, compliance_node):
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("router", router_node)
    workflow.add_node("nurse", nurse_node)
    workflow.add_node("doctor", doctor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("compliance", compliance_node)

    # Router is entry point
    workflow.set_entry_point("router")

    # Conditional routing based on router decision
    def route_after_router(state: OrchestratorState) -> str:
        routes = state.routes or []
        if "Nurse" in routes:
            return "nurse"
        elif "Doctor" in routes:
            return "doctor"
        elif "Research" in routes:
            return "research"
        else:
            return "nurse"  # default fallback

    workflow.add_conditional_edges("router", route_after_router, {
        "nurse": "nurse",
        "doctor": "doctor", 
        "research": "research"
    })

    # After nurse → conditionally go to doctor only if minimal slots satisfied
    def after_nurse(state: OrchestratorState) -> str:
        slots = getattr(state, "slots", {}) or {}
        if slots.get("symptom") and slots.get("duration"):
            return "doctor"
        return "reasoner"

    workflow.add_conditional_edges("nurse", after_nurse, {
        "doctor": "doctor",
        "reasoner": "reasoner",
    })
    # Doctor → Reasoner as usual
    workflow.add_edge("doctor", "reasoner")
    workflow.add_edge("research", "reasoner")

    # Reasoner always → Compliance
    workflow.add_edge("reasoner", "compliance")

    # Compliance → final output
    workflow.set_finish_point("compliance")

    return workflow