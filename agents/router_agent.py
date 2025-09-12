# healthcare_agents/agents/router_agent.py

from typing import List
from services.llm_adapter import LLMAdapter

class RouterAgent:
    """
    Routes user queries to appropriate agents.
    Uses MCP-backed LLMAdapter for reasoning.
    """
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def route(self, query: str, context_messages: List):
        prompt = f"""
        You are a medical triage router.
        Decide which agents should handle this query:
        Options: [Nurse, Doctor, Research].
        Query: {query}
        Context: {context_messages}
        Return a JSON list of agent names.
        """
        response = await self.llm.simple("", prompt)
        try:
            import json
            return json.loads(response)
        except:
            return ["Nurse"]  # safe fallback