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
        """
        Retrieve disease–symptom triples related to query q.
        Tries label-specific schema first, then falls back to a generic schema.
        """
        if not q:
            return []
        try:
            with self.driver.session() as session:
                # Primary: opinionated medical schema
                cypher_primary = """
                MATCH (s:Disease)-[p:IS_SYMPTOM]->(o:Symptom)
                WHERE toLower(o.name) CONTAINS toLower($q)
                RETURN s.name AS s, type(p) AS p, o.name AS o
                LIMIT $limit
                """
                rows = list(session.run(cypher_primary, q=q, limit=limit))
                if not rows:
                    # Fallback: generic entity/relationship schema
                    cypher_fallback = """
                    MATCH (s)-[p]->(o)
                    WHERE o.name IS NOT NULL AND toLower(o.name) CONTAINS toLower($q)
                    RETURN coalesce(s.name, toString(s)) AS s, type(p) AS p, coalesce(o.name, toString(o)) AS o
                    LIMIT $limit
                    """
                    rows = list(session.run(cypher_fallback, q=q, limit=limit))
                result: List[Tuple[str, str, str]] = []
                for r in rows:
                    try:
                        s = str(r.get("s"))
                        p = str(r.get("p"))
                        o = str(r.get("o"))
                        print(f"[KG] {s} -[{p}]-> {o}")
                        result.append((s, p, o))
                    except Exception:
                        continue
                return result
        except Exception as e:
            print(f"[KG] retrieve_triples error for q='{q}': {e}")
            return []

    def find_similar_symptoms(self, target_symptoms: List[str]) -> List[str]:
        """
        Find symptoms in the database that are similar to the target symptoms.
        This helps when exact matches don't exist.
        """
        if not target_symptoms:
            return []
        
        try:
            with self.driver.session() as session:
                # Try to find symptoms that contain any of the target terms
                cypher = """
                MATCH (s:Symptom)
                WHERE any(target IN $targets WHERE toLower(s.name) CONTAINS target)
                RETURN s.name as symptom_name
                ORDER BY s.name
                """
                rows = list(session.run(cypher, targets=target_symptoms))
                return [row['symptom_name'] for row in rows]
        except Exception as e:
            print(f"[KG] find_similar_symptoms error: {e}")
            return []

    def retrieve_diseases_with_all_symptoms(self, symptoms: List[str], limit: int = 100) -> List[Tuple[str,str,str]]:
        """
        Retrieve diseases that have relationships with ALL the provided symptoms.
        This method finds diseases that are connected to every symptom in the list.
        
        Args:
            symptoms: List of symptom names to search for
            limit: Maximum number of results to return
            
        Returns:
            List of tuples (disease, relationship, symptom) for diseases that have ALL symptoms
        """
        if not symptoms:
            return []
        
        # Clean and normalize symptoms
        clean_symptoms = [s.strip().lower() for s in symptoms if s.strip()]
        if not clean_symptoms:
            return []
        
        # First try to find similar symptoms in the database
        similar_symptoms = self.find_similar_symptoms(clean_symptoms)
        
        # Use similar symptoms if we found any, otherwise use original
        search_symptoms = similar_symptoms if similar_symptoms else clean_symptoms
        
        try:
            with self.driver.session() as session:
                
                # Primary: opinionated medical schema - find diseases that have ALL symptoms (exact match)
                cypher_primary = """
                MATCH (d:Disease)-[r:IS_SYMPTOM]->(s:Symptom)
                WHERE toLower(s.name) IN $symptoms
                WITH d, collect(DISTINCT s.name) as disease_symptoms
                WHERE size(disease_symptoms) = $symptom_count
                MATCH (d)-[r2:IS_SYMPTOM]->(s2:Symptom)
                WHERE toLower(s2.name) IN $symptoms
                RETURN d.name AS disease, type(r2) AS relationship, s2.name AS symptom
                LIMIT $limit
                """
                
                rows = list(session.run(
                    cypher_primary, 
                    symptoms=search_symptoms, 
                    symptom_count=len(search_symptoms),
                    limit=limit
                ))
                
                if not rows:
                    # Fallback: Use partial matching to find diseases with symptoms that contain our search terms
                    cypher_partial = """
                    MATCH (d:Disease)-[r:IS_SYMPTOM]->(s:Symptom)
                    WHERE any(symptom IN $symptoms WHERE toLower(s.name) CONTAINS symptom)
                    WITH d, collect(DISTINCT s.name) as disease_symptoms
                    WHERE size(disease_symptoms) >= 2
                    MATCH (d)-[r2:IS_SYMPTOM]->(s2:Symptom)
                    WHERE any(symptom IN $symptoms WHERE toLower(s2.name) CONTAINS symptom)
                    RETURN d.name AS disease, type(r2) AS relationship, s2.name AS symptom
                    LIMIT $limit
                    """
                    rows = list(session.run(
                        cypher_partial, 
                        symptoms=search_symptoms, 
                        limit=limit
                    ))
                    
                    if not rows:
                        # Final fallback: generic entity/relationship schema
                        cypher_fallback = """
                        MATCH (d)-[r]->(s)
                        WHERE s.name IS NOT NULL AND toLower(s.name) IN $symptoms
                        WITH d, collect(DISTINCT s.name) as disease_symptoms
                        WHERE size(disease_symptoms) = $symptom_count
                        MATCH (d)-[r2]->(s2)
                        WHERE s2.name IS NOT NULL AND toLower(s2.name) IN $symptoms
                        RETURN d.name AS disease, type(r2) AS relationship, s2.name AS symptom
                        LIMIT $limit
                        """
                        rows = list(session.run(
                            cypher_fallback, 
                            symptoms=search_symptoms, 
                            symptom_count=len(search_symptoms),
                            limit=limit
                        ))
                
                result: List[Tuple[str, str, str]] = []
                for r in rows:
                    try:
                        disease = str(r.get("disease"))
                        relationship = str(r.get("relationship"))
                        symptom = str(r.get("symptom"))
                        print(f"[KG] {disease} -[{relationship}]-> {symptom}")
                        result.append((disease, relationship, symptom))
                    except Exception:
                        continue
                return result
                
        except Exception as e:
            print(f"[KG] retrieve_diseases_with_all_symptoms error for symptoms={symptoms}: {e}")
            return []