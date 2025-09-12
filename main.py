# main.py
import asyncio, uuid
from orchestrator.orchestrator import Orchestrator

async def demo():
    orch = Orchestrator()
    thread = str(uuid.uuid4())
    print("Begin demo conversation. Type 'exit' to quit.")
    history=[]
    while True:
        u = input("You: ")
        if u.strip().lower() in ("exit","quit"): break
        out = await orch.run_turn(thread, u, prior_messages=history)
        print("Assistant:", out.get("text") or out.get("answer"))
        # append to history
        history.append({"role":"user","text":u})
        history.append({"role":"assistant","text": out.get("text") or out.get("answer")})

if __name__ == "__main__":
    asyncio.run(demo())
