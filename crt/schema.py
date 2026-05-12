"""
schema.py – Shared constants imported by all RES-AI modules.
Centralising ENTITY_SCHEMA here prevents circular imports between
main.py ↔ taxonomy.py ↔ api.py.
"""

from __future__ import annotations
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# NEO4J ENTITY SCHEMA
# Maps every GLiNER label → (Neo4j NodeLabel, RelationshipType)
# Used by: main.py, taxonomy.py, api.py, query_engine.py
# ---------------------------------------------------------------------------
ENTITY_SCHEMA: Dict[str, Tuple[str, str]] = {
    "Programming Language": ("ProgrammingLanguage",  "KNOWS_LANGUAGE"),
    "Framework":            ("Framework",             "USES_FRAMEWORK"),
    "Database":             ("Database",              "USES_DATABASE"),
    "Cloud Platform":       ("CloudPlatform",         "USES_CLOUD_PLATFORM"),
    "Security Tool":        ("SecurityTool",          "USES_TOOL"),
    "Soft Skill":           ("SoftSkill",             "HAS_SOFT_SKILL"),
    "Job Title":            ("JobTitle",              "HAS_TITLE"),
    "Degree":               ("Degree",                "HOLDS_DEGREE"),
    "University":           ("University",            "ATTENDED_UNIVERSITY"),
    "Certification":        ("Certification",         "HAS_CERTIFICATION"),
    "Years of Experience":  ("ExperienceLevel",       "HAS_EXPERIENCE"),
    "Industry":             ("Industry",              "WORKED_IN_INDUSTRY"),
}
