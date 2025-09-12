# services/vdb_service.py
import os, pickle, yaml
from typing import List, Tuple
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..","config", "config.yaml")
CFG = yaml.safe_load(open(config_path,"r",encoding="utf-8"))

class VDBService:
    def __init__(self, index_file: str = None, dim: int = None, model_name: str = None):
        self.index_file = index_file or CFG["faiss"]["general_index"]
        self.dim = dim or CFG["faiss"]["dim"]
        self.model_name = model_name or CFG["faiss"]["embedding_model"]
        os.makedirs(os.path.dirname(self.index_file) or ".", exist_ok=True)
        self.model = SentenceTransformer(self.model_name)
        self.texts = [ "Aspirin is commonly used to treat headaches and mild pain.",
        "Ibuprofen reduces inflammation and helps with arthritis.",
        "Paracetamol is effective in reducing fever.",
        "Metformin is prescribed for type 2 diabetes management.",
        "Amoxicillin is an antibiotic used to treat bacterial infections."]
        if os.path.exists(self.index_file) and os.path.exists(self.index_file + ".meta"):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.index_file + ".meta","rb") as f:
                    self.texts = pickle.load(f)
                if self.index.ntotal != len(self.texts):
                    self.index = faiss.IndexFlatL2(self.dim); self.texts = []
            except Exception:
                self.index = faiss.IndexFlatL2(self.dim); self.texts = []
        else:
            self.index = faiss.IndexFlatL2(self.dim)

    def encode(self, texts: List[str]):
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype("float32")

    def add_chunks(self, texts: List[str], persist: bool = True):
        if not texts: return
        embs = self.encode(texts)
        self.index.add(embs)
        self.texts.extend(texts)
        if persist:
            faiss.write_index(self.index, self.index_file)
            with open(self.index_file + ".meta","wb") as f:
                pickle.dump(self.texts, f)

    def query(self, q: str, top_k: int = None) -> List[Tuple[str, float]]:
        top_k = top_k or CFG["faiss"]["top_k"]
        if self.index.ntotal == 0:
            return []
        emb = self.encode([q])
        D,I = self.index.search(emb, min(top_k, max(1, self.index.ntotal)))
        out=[]
        for i, idx in enumerate(I[0]):
            if idx >=0 and idx < len(self.texts):
                out.append((self.texts[idx], float(D[0][i])))
        return out

    def count(self): return self.index.ntotal
