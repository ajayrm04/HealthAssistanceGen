# services/llm_adapter.py
import yaml, os
from typing import List, Dict, Any, Optional

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..","config", "config.yaml")
cfg = yaml.safe_load(open(config_path,"r",encoding="utf-8"))
PROVIDER = cfg["llm"]["provider"].lower()

# Import provider adapters lazily
if PROVIDER == "ollama":
    from langchain_ollama import ChatOllama as ChatModel
elif PROVIDER == "openai":
    from langchain_openai import ChatOpenAI as ChatModel
# elif PROVIDER == "anthropic":
#     from langchain_anthropic import ChatAnthropic as ChatModel
else:
    raise RuntimeError(f"Unsupported LLM provider: {PROVIDER}")

class LLMAdapter:
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        llm_conf = cfg["llm"]
        model = model_name or llm_conf.get("model_name")
        temp = temperature if temperature is not None else llm_conf.get("temperature", 0.0)
        self.max_tokens = max_tokens or llm_conf.get("max_tokens", 512)
        # instantiate provider-specific chat model
        if PROVIDER == "openai":
            # Try to get API key from config first, then environment variable
            api_key = llm_conf.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("api key required. Set 'openai_api_key' in config.yaml or OPENAI_API_KEY environment variable")
            # langchain_openai.ChatOpenAI accepts api_key and model
            self.client = ChatModel(model=model, temperature=temp, api_key=api_key)
        elif PROVIDER == "ollama":
            # Get Ollama base URL from config or use default
            base_url = llm_conf.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.client = ChatModel(model=model, temperature=temp, base_url=base_url)
        else:
            # ChatAnthropic or other providers accept model & temperature
            self.client = ChatModel(model=model, temperature=temp)

    async def agenerate(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role":"system"/"user"/"assistant", "content": "..."}]
        returns string
        """
        # Use async invoke if available
        resp = await self.client.ainvoke(messages)
        return resp.content  # type: ignore

    async def simple(self, system_prompt: str, user_prompt: str) -> str:
        return await self.agenerate([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}])
    async def run(self, prompt: str) -> str:
        """Backward-compatible wrapper for agenerate."""
        return await self.agenerate([{"role": "user", "content": prompt}])
