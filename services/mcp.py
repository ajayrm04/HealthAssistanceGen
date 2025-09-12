# services/mcp.py
import hashlib, json, os, math
from typing import List, Dict, Any, Tuple
import tiktoken

def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

def approx_tokens(text: str, enc_name: str = "cl100k_base") -> int:
    try:
        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(text))
    except Exception:
        return max(1, math.ceil(len(text)/4))

class MCPAssembler:
    def __init__(self, max_tokens: int = 1600, max_items: int = 16, prefer_kg_boost: float = 1.2):
        self.max_tokens = max_tokens
        self.max_items = max_items
        self.prefer_kg_boost = prefer_kg_boost

    def from_vdb(self, vdb_results: List[Tuple[str, float]], source: str = "faiss"):
        out=[]
        for i,(txt, score) in enumerate(vdb_results):
            uid = make_id(txt + source + str(i))
            out.append({"id":f"VDB#{uid}", "type":"vdb", "source":source, "content":txt, "meta":{"score":float(score)}, "tokens": approx_tokens(txt)})
        return out

    def from_kg(self, kg_triples: List[Tuple[str,str,str]], source: str = "neo4j"):
        out=[]
        for i,(s,p,o) in enumerate(kg_triples):
            txt = f"({s}) -[{p}]-> ({o})"
            uid = make_id(txt + source + str(i))
            out.append({"id":f"KG#{uid}", "type":"kg", "source":source, "content":txt, "meta":{"score":1.0}, "tokens": approx_tokens(txt)})
        return out

    def dedupe_and_rank(self, items: List[Dict[str,Any]]):
        seen={}
        for ev in items:
            k = ev["content"].strip().lower()
            if k in seen:
                if ev["meta"].get("score",0) > seen[k]["meta"].get("score",0):
                    seen[k]=ev
            else:
                seen[k]=ev
        uniq = list(seen.values())
        for ev in uniq:
            ev["meta"]["adj"] = ev["meta"].get("score",0) * (self.prefer_kg_boost if ev["type"]=="kg" else 1.0)
        ranked = sorted(uniq, key=lambda x: (x["meta"]["adj"], -x["tokens"]), reverse=True)
        return ranked

    def assemble_context(self, evidences: List[Dict[str,Any]], question: str):
        selected = []
        total = approx_tokens(question)
        for ev in evidences:
            if len(selected) >= self.max_items: break
            toks = ev.get("tokens", approx_tokens(ev.get("content","")))
            if total + toks > self.max_tokens and len(selected) > 0:
                break
            selected.append(ev)
            total += toks
        stats = {"requested_max_tokens": self.max_tokens, "used_tokens_approx": total, "selected_count": len(selected)}
        return {"question":question, "evidence": selected, "stats": stats}

    def persist_mcp(self, payload: Dict[str,Any], path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
