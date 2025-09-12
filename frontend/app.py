import sys
import os

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# frontend/app.py
import streamlit as st, yaml, asyncio, uuid
from services.vdb_service import VDBService
from services.kg_service import KGService
from orchestrator.orchestrator import Orchestrator
from services.a2a import A2AClient
from services.slot_extractor import SlotExtractor
from services.mcp import MCPAssembler
from services.reasoner import MCPReasoner
from langchain_core.messages import HumanMessage, AIMessage  # ✅ import
import nest_asyncio
nest_asyncio.apply()

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..", "config", "config.yaml")
CFG = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

st.set_page_config(page_title=CFG["app"]["ui_title"], layout="centered")
st.title(CFG["app"]["ui_title"])
st.markdown(
    "**Disclaimer:** Prototype system. Not a substitute for professional medical advice. Always consult a clinician."
)

@st.cache_resource
def init_system():
    vdb = VDBService()
    kg = KGService()
    orch = Orchestrator()
    return {"kg": kg, "orch": orch}

SYS = init_system()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("You:", key="user_input")
with col2:
    if st.button("Reset session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

if st.button("Send") and user_input:
    # Add user message as HumanMessage
    st.session_state.history.append(HumanMessage(content=user_input))

    # Run orchestrator turn
    res = asyncio.run(
        SYS["orch"].run_turn(
            st.session_state.thread_id,
            user_input,
            prior_messages=st.session_state.history,
        )
    )

    # Final assistant response
    out_text = res["text"]

# Add assistant message
    st.session_state.history.append(AIMessage(content=out_text))

    # Show compliance info if any
    comp = res["raw"].get("compliance")
    if comp:
        st.write("**Compliance outcome:**", comp)

    st.rerun()

st.subheader("Conversation")
for m in st.session_state.history:
    if isinstance(m, HumanMessage):
        st.markdown(f"**You:** {m.content}")
    elif isinstance(m, AIMessage):
        st.markdown(f"**Assistant:** {m.content}")
    else:
        st.markdown(f"**{m.__class__.__name__}:** {getattr(m,'content',str(m))}")

st.markdown("---")
st.header("Agent / A2A Debug")
if st.button("Show last MCP"):
    st.write("MCP info is available in orchestrator.checkpointer or logs (not shown here).")
