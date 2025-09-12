# healthcare_agents/services/utils.py
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def format_agent_message(agent_name: str, content: str):
    """
    Map custom agent names into supported LangChain message roles.
    Adds the agent tag inside the message content for traceability.
    """
    agent_name = agent_name.lower()

    if agent_name in ["doctor", "nurse", "research"]:
        return AIMessage(content=f"[{agent_name.title()}] {content}")

    elif agent_name == "compliance":
        return SystemMessage(content=f"[Compliance] {content}")

    else:
        return AIMessage(content=f"[{agent_name}] {content}")