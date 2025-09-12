# healthcare_agents/orchestrator/state_schema.py

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from typing_extensions import Annotated
try:
    from langgraph.graph.message import add_messages
except ImportError:
    # Fallback if langgraph isn't available
    def add_messages(messages, new_messages):
        return messages + new_messages
from langchain_core.messages import BaseMessage

@dataclass
class OrchestratorState:
    """
    Shared conversation state passed through the orchestrator graph.
    """

    # Conversation/session tracking
    thread_id: str

    # Latest user query
    user_query: str

    # Message history: (speaker, message)
    # messages: Annotated[List[Tuple[str, str]], add_messages] = field(default_factory=list)
    
    messages: Annotated[List[BaseMessage], add_messages]
    slots: Dict[str, Any] = field(default_factory=dict)
    # Router decisions (list of agent names that should be called)
    routes: Optional[List[str]] = None

    # Final output after compliance gating
    final_response: Optional[str] = None
