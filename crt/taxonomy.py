"""
Phase 3 – taxonomy.py
=====================
• SKILL_TAXONOMY  : ESCO-inspired directed skill hierarchy
• SkillTaxonomySeeder : seeds the master graph into Neo4j (idempotent)
• GraphRAGLinker  : links candidate extracted entities → taxonomy nodes
                    so graph traversals work across ALL candidates.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from schema import ENTITY_SCHEMA  # no circular import

logger = logging.getLogger("RES-AI.Taxonomy")

# ---------------------------------------------------------------------------
# MASTER SKILL TAXONOMY
# Format:  child_name -> (parent_name, parent_label, child_label, rel_type)
# ---------------------------------------------------------------------------
# TaxEntry = (child_label, parent_name, parent_label, relationship_type)
TaxEntry = Tuple[str, str, str, str]

SKILL_TAXONOMY: Dict[str, TaxEntry] = {
    # ── Python Ecosystem ────────────────────────────────────────────────
    "Django":        ("Framework",           "Python",           "ProgrammingLanguage", "IS_FRAMEWORK_OF"),
    "Flask":         ("Framework",           "Python",           "ProgrammingLanguage", "IS_FRAMEWORK_OF"),
    "Fastapi":       ("Framework",           "Python",           "ProgrammingLanguage", "IS_FRAMEWORK_OF"),
    "Pytorch":       ("Framework",           "Python",           "ProgrammingLanguage", "IS_LIBRARY_OF"),
    "Tensorflow":    ("Framework",           "Python",           "ProgrammingLanguage", "IS_LIBRARY_OF"),
    "Numpy":         ("Framework",           "Python",           "ProgrammingLanguage", "IS_LIBRARY_OF"),
    "Pandas":        ("Framework",           "Python",           "ProgrammingLanguage", "IS_LIBRARY_OF"),
    "Scikit-Learn":  ("Framework",           "Python",           "ProgrammingLanguage", "IS_LIBRARY_OF"),
    "Celery":        ("Framework",           "Python",           "ProgrammingLanguage", "IS_LIBRARY_OF"),

    # ── JavaScript Ecosystem ─────────────────────────────────────────────
    "React":         ("Framework",           "Javascript",       "ProgrammingLanguage", "IS_FRAMEWORK_OF"),
    "Next.Js":       ("Framework",           "Javascript",       "ProgrammingLanguage", "IS_FRAMEWORK_OF"),
    "Vue":           ("Framework",           "Javascript",       "ProgrammingLanguage", "IS_FRAMEWORK_OF"),
    "Angular":       ("Framework",           "Javascript",       "ProgrammingLanguage", "IS_FRAMEWORK_OF"),
    "Node.Js":       ("Framework",           "Javascript",       "ProgrammingLanguage", "IS_RUNTIME_OF"),
    "Express":       ("Framework",           "Node.Js",          "Framework",           "IS_FRAMEWORK_OF"),

    # ── Java Ecosystem ───────────────────────────────────────────────────
    "Spring":        ("Framework",           "Java",             "ProgrammingLanguage", "IS_FRAMEWORK_OF"),
    "Spring Boot":   ("Framework",           "Java",             "ProgrammingLanguage", "IS_FRAMEWORK_OF"),

    # ── Dart/Flutter ─────────────────────────────────────────────────────
    "Flutter":       ("Framework",           "Dart",             "ProgrammingLanguage", "IS_FRAMEWORK_OF"),

    # ── Databases: SQL family ─────────────────────────────────────────────
    "Postgresql":    ("Database",            "Sql",              "DatabaseCategory",    "IS_TYPE_OF"),
    "Mysql":         ("Database",            "Sql",              "DatabaseCategory",    "IS_TYPE_OF"),
    "Sqlite":        ("Database",            "Sql",              "DatabaseCategory",    "IS_TYPE_OF"),

    # ── Databases: NoSQL family ───────────────────────────────────────────
    "Mongodb":       ("Database",            "Nosql",            "DatabaseCategory",    "IS_TYPE_OF"),
    "Redis":         ("Database",            "Nosql",            "DatabaseCategory",    "IS_TYPE_OF"),
    "Cassandra":     ("Database",            "Nosql",            "DatabaseCategory",    "IS_TYPE_OF"),
    "Elasticsearch": ("Database",            "Nosql",            "DatabaseCategory",    "IS_TYPE_OF"),

    # ── Databases: Graph ─────────────────────────────────────────────────
    "Neo4J":         ("Database",            "Graph Database",   "DatabaseCategory",    "IS_TYPE_OF"),

    # ── Cloud Platforms ───────────────────────────────────────────────────
    "Aws":           ("CloudPlatform",       "Cloud Computing",  "Skill",               "IS_PLATFORM_OF"),
    "Gcp":           ("CloudPlatform",       "Cloud Computing",  "Skill",               "IS_PLATFORM_OF"),
    "Azure":         ("CloudPlatform",       "Cloud Computing",  "Skill",               "IS_PLATFORM_OF"),
    "Firebase":      ("CloudPlatform",       "Cloud Computing",  "Skill",               "IS_PLATFORM_OF"),
    "Supabase":      ("CloudPlatform",       "Cloud Computing",  "Skill",               "IS_PLATFORM_OF"),

    # ── Security Tools ───────────────────────────────────────────────────
    "Wireshark":     ("SecurityTool",        "Network Analysis", "SecurityDomain",      "BELONGS_TO"),
    "Nmap":          ("SecurityTool",        "Network Analysis", "SecurityDomain",      "BELONGS_TO"),
    "Metasploit":    ("SecurityTool",        "Penetration Testing", "SecurityDomain",   "BELONGS_TO"),
    "Burp Suite":    ("SecurityTool",        "Penetration Testing", "SecurityDomain",   "BELONGS_TO"),
    "Splunk":        ("SecurityTool",        "SIEM",             "SecurityDomain",      "BELONGS_TO"),
    "Elastic Siem":  ("SecurityTool",        "SIEM",             "SecurityDomain",      "BELONGS_TO"),
    "Snort":         ("SecurityTool",        "Intrusion Detection", "SecurityDomain",   "BELONGS_TO"),
    "Suricata":      ("SecurityTool",        "Intrusion Detection", "SecurityDomain",   "BELONGS_TO"),

    # ── DevOps ───────────────────────────────────────────────────────────
    "Docker":        ("Skill",               "DevOps",           "Skill",               "BELONGS_TO_DOMAIN"),
    "Kubernetes":    ("Skill",               "DevOps",           "Skill",               "BELONGS_TO_DOMAIN"),
    "Jenkins":       ("Skill",               "Ci/Cd",            "Skill",               "IS_TOOL_OF"),
    "Github Actions":("Skill",              "Ci/Cd",            "Skill",               "IS_TOOL_OF"),
    "Terraform":     ("Skill",               "Infrastructure As Code", "Skill",         "IS_TOOL_OF"),
    "Ansible":       ("Skill",               "Infrastructure As Code", "Skill",         "IS_TOOL_OF"),

    # ── AI / ML Domains ──────────────────────────────────────────────────
    "Machine Learning":    ("Skill",         "Artificial Intelligence", "Skill",        "IS_SUBDOMAIN_OF"),
    "Deep Learning":       ("Skill",         "Machine Learning",       "Skill",         "IS_SUBDOMAIN_OF"),
    "Natural Language Processing": ("Skill", "Artificial Intelligence", "Skill",        "IS_SUBDOMAIN_OF"),
    "Computer Vision":     ("Skill",         "Artificial Intelligence", "Skill",        "IS_SUBDOMAIN_OF"),
    "Llm":                 ("Skill",         "Natural Language Processing", "Skill",    "IS_SUBDOMAIN_OF"),

    # ── Security Domains ─────────────────────────────────────────────────
    "Penetration Testing": ("Skill",         "Cyber Security",   "Skill",               "IS_SUBDOMAIN_OF"),
    "Network Analysis":    ("Skill",         "Cyber Security",   "Skill",               "IS_SUBDOMAIN_OF"),
    "Blue Team":           ("Skill",         "Cyber Security",   "Skill",               "IS_SUBDOMAIN_OF"),
    "Red Team":            ("Skill",         "Cyber Security",   "Skill",               "IS_SUBDOMAIN_OF"),
    "Siem":                ("Skill",         "Cyber Security",   "Skill",               "IS_SUBDOMAIN_OF"),
    "Soc":                 ("Skill",         "Cyber Security",   "Skill",               "IS_SUBDOMAIN_OF"),
    "Intrusion Detection": ("Skill",         "Cyber Security",   "Skill",               "IS_SUBDOMAIN_OF"),
}


# ---------------------------------------------------------------------------
# SKILL TAXONOMY SEEDER
# ---------------------------------------------------------------------------
class SkillTaxonomySeeder:
    """
    Seeds SKILL_TAXONOMY into Neo4j as a master knowledge graph.
    Safe to call multiple times — all writes use MERGE (idempotent).

    Graph pattern created:
        (child:{child_label} {name})-[:rel_type]->(parent:{parent_label} {name})
    """

    def __init__(self, driver) -> None:
        self.driver = driver
        self._log = logging.getLogger("RES-AI.TaxonomySeeder")

    def seed(self) -> int:
        """Seed the taxonomy. Returns number of relationships created/merged."""
        self._log.info("Seeding master skill taxonomy (%d entries)…", len(SKILL_TAXONOMY))
        count = 0
        with self.driver.session() as session:
            for child_name, (child_label, parent_name, parent_label, rel_type) in SKILL_TAXONOMY.items():
                session.run(
                    f"""
                    MERGE (parent:{parent_label} {{name: $parent_name}})
                      ON CREATE SET parent.taxonomy = true, parent.created_at = timestamp()
                    MERGE (child:{child_label} {{name: $child_name}})
                      ON CREATE SET child.taxonomy = true, child.created_at = timestamp()
                    MERGE (child)-[r:{rel_type}]->(parent)
                      ON CREATE SET r.source = 'ESCO_TAXONOMY'
                    """,
                    parent_name=parent_name,
                    child_name=child_name,
                )
                count += 1
        self._log.info("Taxonomy seeded: %d skill relationships written.", count)
        return count


# ---------------------------------------------------------------------------
# GRAPHRAG LINKER
# ---------------------------------------------------------------------------
class GraphRAGLinker:
    """
    After a candidate is written to Neo4j (Phase 2), this linker connects
    the candidate's entity nodes to the master taxonomy graph using
    SIMILAR_TO relationships.

    This enables GraphRAG queries like:
        "Find candidates who know the Python ecosystem"
    which traverses:
        Candidate → FastAPI → IS_FRAMEWORK_OF → Python
    """

    def __init__(self, driver) -> None:
        self.driver = driver
        self._log = logging.getLogger("RES-AI.GraphRAGLinker")

    def link_candidate(self, candidate_id: str, by_category: Dict[str, List[Dict]]) -> int:
        """
        For every extracted entity, if a taxonomy node with the same name exists,
        create a SIMILAR_TO edge from the candidate's entity node to it.
        Uses a single UNWIND batch query per category — no N+1 transactions.
        Returns total number of taxonomy edges created.
        """
        self._log.info("Linking candidate %s to taxonomy…", candidate_id)

        # Flatten all extracted entity names into one list
        names: List[str] = []
        for gliner_label, items in by_category.items():
            if gliner_label in ENTITY_SCHEMA:
                names.extend(item["text"] for item in items)

        if not names:
            return 0

        count = 0
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $names AS name
                MATCH (entity {name: name})
                WHERE NOT entity.taxonomy IS NOT NULL
                MATCH (tax {name: name, taxonomy: true})
                WHERE tax <> entity
                MERGE (entity)-[:SIMILAR_TO]->(tax)
                RETURN count(*) AS total
                """,
                names=names,
            )
            rec = result.single()
            count = rec["total"] if rec else 0

        self._log.info("GraphRAG linking complete: %d taxonomy edges for %s.", count, candidate_id)
        return count
