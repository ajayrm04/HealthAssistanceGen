# services/kg_service.py
from typing import List, Tuple, Optional
from neo4j import GraphDatabase, basic_auth
import yaml
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..","config", "config.yaml")
CFG = yaml.safe_load(open(config_path,"r",encoding="utf-8"))

class KGService:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        uri = uri or CFG["neo4j"]["uri"]
        user = user or CFG["neo4j"]["user"]
        password = password or CFG["neo4j"]["password"]
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    def close(self):
        self.driver.close()

    def insert_triples(self, triples: List[Tuple[str,str,str]]):
        if not triples: return
        cypher = """
        UNWIND $rows AS row
        MERGE (s:Entity {name:row.s})
        MERGE (o:Entity {name:row.o})
        MERGE (s)-[r:REL {type:row.p}]->(o)
        """
        rows = [{"s":s, "p":p, "o":o} for s,p,o in triples]
        with self.driver.session() as session:
            session.run(cypher, rows=rows)

    def retrieve_triples(self, q: str, limit: int = 20) -> List[Tuple[str,str,str]]:
        cypher = """
        MATCH (s:Disease)-[p:IS_SYMPTOM]->(o:Symptom)
        WHERE toLower(o.name) IN $q
        RETURN s.name AS s, type(p) AS p, o.name AS o
        LIMIT $limit
        """
        with self.driver.session() as session:
            res = session.run(cypher, q=q, limit=limit)
            rows = list(res)
            for r in rows:
                try:
                    print(r["s"], r["p"], r["o"])  # debug
                except Exception:
                    pass
            return [(r["s"], r["p"], r["o"]) for r in rows]
