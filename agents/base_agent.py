# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
ROLE_MAP = {
    "doctor": "assistant",
    "nurse": "assistant",
    "researcher": "assistant",
    "patient": "user",
    "system": "system",
    "assistant": "assistant",
    "user": "user"
}
class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.memory: List[Dict[str,Any]] = []

    def remember(self, role: str, text: str):
        mapped_role = ROLE_MAP.get(role, "assistant")
        self.memory.append({"type": mapped_role, "content": text})

    def recent(self, n: int = 10):
        return self.memory[-n:]

    @abstractmethod
    async def handle(self, state: Dict[str,Any]) -> Dict[str,Any]:
        raise NotImplementedError
