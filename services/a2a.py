# services/a2a.py
from typing import Callable, Dict, Any, Optional
import uuid, time
try:
    import httpx
except Exception:
    httpx = None

class LocalA2ABus:
    def __init__(self):
        self.registry: Dict[str, Callable[[Dict[str,Any]], Dict[str,Any]]] = {}

    def register(self, name: str, handler: Callable[[Dict[str,Any]], Dict[str,Any]]):
        self.registry[name] = handler

    def send(self, envelope: Dict[str,Any]) -> Dict[str,Any]:
        to = envelope.get("to")
        if to not in self.registry:
            return {"status":"error", "error": f"target {to} not registered"}
        return self.registry[to](envelope)

class A2AClient:
    def __init__(self, enabled: bool = True, transport: str = "local", http_endpoint: Optional[str] = None):
        self.enabled = enabled
        self.transport = transport
        self.http_endpoint = http_endpoint
        self.local = LocalA2ABus()

    def register_local(self, name: str, handler: Callable[[Dict[str,Any]], Dict[str,Any]]):
        self.local.register(name, handler)

    def _envelope(self, frm: str, to: str, capability: str, thread_id: str, payload: Dict[str,Any]):
        return {"id": str(uuid.uuid4()), "ts": int(time.time()), "from":frm, "to":to, "capability":capability, "thread_id":thread_id, "payload":payload}

    def send(self, frm: str, to: str, capability: str, thread_id: str, payload: Dict[str,Any]):
        if not self.enabled:
            return {"status":"disabled"}
        env = self._envelope(frm, to, capability, thread_id, payload)
        if self.transport == "local":
            return self.local.send(env)
        elif self.transport == "http":
            if httpx is None:
                return {"status":"error","error":"httpx not installed"}
            try:
                resp = httpx.post(self.http_endpoint, json=env, timeout=20.0)
                return resp.json()
            except Exception as e:
                return {"status":"error","error": str(e)}
        return {"status":"error","error":"unsupported transport"}
