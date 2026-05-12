from neo4j import GraphDatabase
import json

class ResumeGraphBuilder:
    def __init__(self, uri, user, password):
        print("[*] Connecting to Neo4j Database...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Clear the database for testing (WARNING: Drops all existing data)
        self._clear_database()

    def close(self):
        self.driver.close()

    def _clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print(" └── Database cleared for fresh ingestion.")

    def ingest_candidate_data(self, candidate_name: str, extracted_json: dict):
        """Creates the candidate node and links them to all extracted skills."""
        print(f"[*] Ingesting graph data for candidate: {candidate_name}")
        
        with self.driver.session() as session:
            # 1. Create the Candidate Node
            session.execute_write(self._create_candidate, candidate_name)
            
            # 2. Iterate through the JSON and create Skill Nodes + Relationships
            for category, items in extracted_json.items():
                for item in items:
                    session.execute_write(self._create_skill_link, candidate_name, category, item)
            
            print(" └── Graph construction complete!")

    @staticmethod
    def _create_candidate(tx, name):
        # Cypher query to create a Candidate
        query = "MERGE (c:Candidate {name: $name})"
        tx.run(query, name=name)

    @staticmethod
    def _create_skill_link(tx, candidate_name, category, skill_name):
        # Cypher query to create the Skill node and the HAS_SKILL relationship
        # We replace spaces with underscores for the node label (e.g., 'Programming Language' -> 'Programming_Language')
        label = category.replace(" ", "_")
        
        query = f"""
        MATCH (c:Candidate {{name: $candidate_name}})
        MERGE (s:{label} {{name: $skill_name}})
        MERGE (c)-[:HAS_SKILL]->(s)
        """
        tx.run(query, candidate_name=candidate_name, skill_name=skill_name)

# --- Execution Block ---
if __name__ == "__main__":
    # The exact JSON output from your terminal
    sample_extracted_data = {
        "Programming Language": ["JavaScript", "Python", "PyTorch", "Node.js", "C/C++", "Dart", "Java"],
        "Database": ["MongoDB", "Redis", "MySQL", "PostgreSQL"],
        "Cloud Platform": ["Firebase", "MinIO", "Supabase"],
        "Security Tool": ["Cisco Packet Tracer", "Wireshark", "Splunk", "Metasploit", "Nmap"],
        "Job Title": ["Cyber Security Intern", "2027 passout"],
        "Degree": ["B.Tech"],
        "University": ["Amrita Vishwa Vidyapeetham"]
    }

    # Connect to the local Docker container
    builder = ResumeGraphBuilder("bolt://localhost:7687", "neo4j", "supersecretpassword")
    
    try:
        # We use a generic ID for the candidate since PII was scrubbed
        builder.ingest_candidate_data("Candidate_001", sample_extracted_data)
    finally:
        builder.close()