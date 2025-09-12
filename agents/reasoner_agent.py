# healthcare_agents/agents/reasoner_agent.py

from typing import List
from services.llm_adapter import LLMAdapter

class ReasonerAgent:
    """
    Fuses multi-agent outputs (Doctor, Research, Nurse).
    Performs reasoning, resolves conflicts, before ComplianceAgent sees them.
    """
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def reason(self, messages: List):
        # KG-only mode: extract any KG findings from the doctor agent message if present
        # and pass through without adding LLM knowledge. If none found, keep minimal echo.
        try:
            text_blobs = [getattr(m, "content", str(m)) for m in (messages or [])]
            kg_blocks = [t for t in text_blobs if "KG findings for symptom" in str(t)]
            if kg_blocks:
                prompt = f"use the knowledge graph data which is {kg_blocks} to reason about the patient's condition . make sure to print the disease names and their symptoms in the output"
                return await self.llm.simple("", prompt)
            # Fallback minimal pass-through
            prompt =f'''
                You are the medical reasoning agent.
                Review the following multi-agent outputs:
                {messages}
                
                - Resolve conflicts if present
                - Add differential reasoning
                - Prepare a single coherent draft answer
                - Do NOT give final advice; that goes through compliance

                Return your reasoning draft as plain text.
         '''
            smtg = await self.llm.simple("", prompt)
            # print(smtg)
            return smtg
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
