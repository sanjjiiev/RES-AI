"""
Phase 3 – query_engine.py
==========================
GraphRAG-style query engine for the RES-AI knowledge graph.
All queries use multi-hop graph traversal — going through the taxonomy
lets us match candidates even when their exact skill text differs from
the search term (e.g. "FastAPI" matches candidates who know "Python").
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RES-AI.QueryEngine")


class GraphQueryEngine:
    """
    Production-grade query engine wrapping Neo4j graph traversals.
    Every method returns plain Python dicts — easy to serialise for APIs.
    """

    def __init__(self, driver) -> None:
        self.driver = driver
        self._log = logging.getLogger("RES-AI.QueryEngine")

    # ------------------------------------------------------------------
    # Candidate profile
    # ------------------------------------------------------------------

    def get_candidate_profile(self, candidate_id: str) -> Optional[Dict]:
        """Return every entity linked to a candidate, grouped by category."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Candidate {id: $id})-[r]->(n)
                RETURN
                    c.id            AS candidate_id,
                    c.file_name     AS file_name,
                    c.ingested_at   AS ingested_at,
                    type(r)         AS relationship,
                    labels(n)[0]    AS category,
                    n.name          AS skill,
                    r.confidence    AS confidence
                ORDER BY category, confidence DESC
                """,
                id=candidate_id,
            )
            rows = [dict(r) for r in result]

        if not rows:
            return None

        profile: Dict[str, Any] = {
            "candidate_id": rows[0]["candidate_id"],
            "file_name":    rows[0]["file_name"],
            "ingested_at":  rows[0]["ingested_at"],
            "skills":       {},
        }
        for row in rows:
            cat = row["category"]
            profile["skills"].setdefault(cat, []).append(
                {"name": row["skill"], "confidence": row["confidence"]}
            )
        return profile

    # ------------------------------------------------------------------
    # Find by exact skill name
    # ------------------------------------------------------------------

    def find_by_skill(self, skill_name: str) -> List[Dict]:
        """Direct match: candidates who have an entity node named skill_name."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Candidate)-[r]->(n {name: $name})
                RETURN
                    c.id          AS candidate_id,
                    c.file_name   AS file,
                    type(r)       AS relationship,
                    r.confidence  AS confidence
                ORDER BY confidence DESC
                """,
                name=skill_name,
            )
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # GraphRAG ecosystem traversal
    # ------------------------------------------------------------------

    def find_by_ecosystem(self, root_skill: str, max_hops: int = 2) -> List[Dict]:
        """
        Multi-hop traversal: find candidates who know root_skill OR anything
        that is_framework_of / is_library_of / is_runtime_of root_skill
        up to max_hops levels deep through the taxonomy.
        """
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (root {{name: $root}})
                MATCH (related)-[:IS_FRAMEWORK_OF|IS_LIBRARY_OF|IS_RUNTIME_OF|IS_SUBDOMAIN_OF*1..{max_hops}]->(root)
                WITH collect(related.name) + [$root] AS ecosystem
                MATCH (c:Candidate)-[r]->(n)
                WHERE n.name IN ecosystem
                RETURN
                    c.id         AS candidate_id,
                    c.file_name  AS file,
                    collect(DISTINCT n.name) AS matched_skills,
                    count(n)     AS match_count
                ORDER BY match_count DESC
                """,
                root=root_skill,
            )
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Multi-skill AND match (all skills required)
    # ------------------------------------------------------------------

    def find_by_skills_all(self, skill_names: List[str]) -> List[Dict]:
        """Return candidates who have ALL of the listed skills."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Candidate)-[r]->(n)
                WHERE n.name IN $skills
                WITH c, collect(DISTINCT n.name) AS matched, count(DISTINCT n) AS cnt
                WHERE cnt = size($skills)
                RETURN c.id AS candidate_id, c.file_name AS file, matched
                ORDER BY c.id
                """,
                skills=skill_names,
            )
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # JD-based candidate ranking
    # ------------------------------------------------------------------

    def rank_candidates_for_jd(
        self,
        required_skills: List[str],
        nice_to_have: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Score candidates based on how many required (weight=2) and
        nice-to-have (weight=1) skills they match. Returns ranked list.
        """
        nice_to_have = nice_to_have or []
        all_skills   = required_skills + nice_to_have

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Candidate)-[r]->(n)
                WHERE n.name IN $all_skills
                WITH c,
                     [x IN collect(n.name) WHERE x IN $required] AS req_matched,
                     [x IN collect(n.name) WHERE x IN $nice]      AS nice_matched
                RETURN
                    c.id          AS candidate_id,
                    c.file_name   AS file,
                    req_matched,
                    nice_matched,
                    size(req_matched)*2 + size(nice_matched) AS score
                ORDER BY score DESC
                LIMIT $top_n
                """,
                all_skills=all_skills,
                required=required_skills,
                nice=nice_to_have,
                top_n=top_n,
            )
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def graph_stats(self) -> Dict:
        """Return high-level counts for dashboard display."""
        with self.driver.session() as session:
            stats = {}
            stats["candidates"] = session.run(
                "MATCH (c:Candidate) RETURN count(c) AS n"
            ).single()["n"]
            stats["skill_nodes"] = session.run(
                "MATCH (n) WHERE NOT n:Candidate RETURN count(n) AS n"
            ).single()["n"]
            stats["relationships"] = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS n"
            ).single()["n"]
            stats["taxonomy_nodes"] = session.run(
                "MATCH (n {taxonomy: true}) RETURN count(n) AS n"
            ).single()["n"]
            return stats

    # ------------------------------------------------------------------
    # Security-domain specific (Blue Team / SOC use-case from doc)
    # ------------------------------------------------------------------

    def find_security_candidates(
        self,
        domains: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Find candidates who work in specific security domains AND know
        specific programming languages — the exact GraphRAG use-case
        described in the RES-AI spec.
        """
        domains   = domains   or ["Blue Team", "Soc", "Siem", "Penetration Testing"]
        languages = languages or ["Python", "Javascript"]

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Candidate)-[:HAS_SOFT_SKILL|USES_TOOL]->(sec)
                WHERE sec.name IN $domains
                MATCH (c)-[:KNOWS_LANGUAGE]->(lang)
                WHERE lang.name IN $languages
                RETURN
                    c.id          AS candidate_id,
                    c.file_name   AS file,
                    collect(DISTINCT sec.name)  AS security_domains,
                    collect(DISTINCT lang.name) AS languages
                ORDER BY size(security_domains) DESC
                """,
                domains=domains,
                languages=languages,
            )
            return [dict(r) for r in result]
