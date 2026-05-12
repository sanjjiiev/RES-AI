"""
evaluator.py – Candidate Evaluation Engine  (Phase 4)
=======================================================
Accepts a JDProfile (from jd_parser.py) and ranks all candidates
in the Neo4j knowledge graph against it.

Scoring model
-------------
• Required skill match  : +2 pts each
• Nice-to-have match    : +1 pt each
• Certification match   : +3 pts each (high weight — often mandatory)
• Degree match          : +2 pts each
• Experience coverage   : bonus +2 if candidate has any experience entity
                          and the JD specifies a min_years_experience

Optional: Ollama LLM generates a 3-sentence narrative for each top candidate.
          Falls back silently if Ollama is not running.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

logger = logging.getLogger("RES-AI.Evaluator")


# ---------------------------------------------------------------------------
# Output Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CandidateScore:
    """Scored result for one candidate."""
    rank: int
    candidate_id: str
    file_name: str
    total_score: int
    coverage_pct: float          # required skills covered (0.0 – 1.0)

    required_matched: List[str]
    nice_matched: List[str]
    cert_matched: List[str]
    degree_matched: List[str]
    has_experience_node: bool

    gaps: List[str]              # required skills NOT matched
    llm_summary: Optional[str] = None


@dataclass
class EvaluationResult:
    """Full output of a JD evaluation run."""
    jd_source: str
    jd_job_title: str
    jd_required_skills: List[str]
    jd_nice_to_have: List[str]
    jd_certifications: List[str]
    jd_degrees: List[str]
    jd_min_experience: Optional[int]
    total_candidates_evaluated: int
    ranked_candidates: List[CandidateScore]


# ---------------------------------------------------------------------------
# CANDIDATE EVALUATOR
# ---------------------------------------------------------------------------

class CandidateEvaluator:
    """
    Scores and ranks all candidates in Neo4j against a JDProfile.

    Workflow
    --------
    1. Receive a JDProfile (already parsed from DOCX by JobDescriptionParser)
    2. Run a single graph query that fetches all candidate skill data
    3. Score each candidate in Python (avoids complex Cypher scoring logic)
    4. Optionally call Ollama for a narrative summary of each top-N candidate
    """

    def __init__(self, driver, gliner_model=None) -> None:
        self.driver      = driver
        self._log        = logging.getLogger("RES-AI.Evaluator")
        self._ollama_url = os.getenv("OLLAMA_URL",   "http://localhost:11434")
        self._ollama_mdl = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_all_candidates(self) -> Dict[str, Dict]:
        """
        Single Neo4j query: pull every candidate + all their linked entity names.
        Returns: {candidate_id: {file_name, skills: set, certs: set, degrees: set, has_exp: bool}}
        """
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Candidate)-[r]->(n)
                RETURN
                    c.id        AS candidate_id,
                    c.file_name AS file_name,
                    type(r)     AS rel_type,
                    n.name      AS skill_name
                """
            )
            candidates: Dict[str, Dict] = {}
            for row in rows:
                cid = row["candidate_id"]
                if cid not in candidates:
                    candidates[cid] = {
                        "file_name": row["file_name"] or "",
                        "skills":    set(),
                        "certs":     set(),
                        "degrees":   set(),
                        "has_exp":   False,
                    }
                rel  = row["rel_type"]
                name = row["skill_name"] or ""
                if rel == "HAS_CERTIFICATION":
                    candidates[cid]["certs"].add(name)
                elif rel == "HOLDS_DEGREE":
                    candidates[cid]["degrees"].add(name)
                elif rel == "HAS_EXPERIENCE":
                    candidates[cid]["has_exp"] = True
                else:
                    candidates[cid]["skills"].add(name)
        return candidates

    def _score_candidate(
        self,
        cdata: Dict,
        jd,       # JDProfile
    ) -> tuple[int, List[str], List[str], List[str], List[str], float, List[str]]:
        """
        Score one candidate against the JDProfile.
        Returns: (score, req_matched, nice_matched, cert_matched, deg_matched, coverage, gaps)
        """
        skills  = cdata["skills"]
        certs   = cdata["certs"]
        degrees = cdata["degrees"]
        has_exp = cdata["has_exp"]

        req_matched  = [s for s in jd.required_skills      if s in skills]
        nice_matched = [s for s in jd.nice_to_have_skills   if s in skills]
        cert_matched = [c for c in jd.required_certifications if c in certs]
        deg_matched  = [d for d in jd.required_degrees       if d in degrees]
        gaps         = [s for s in jd.required_skills if s not in skills]

        score  = len(req_matched) * 2
        score += len(nice_matched) * 1
        score += len(cert_matched) * 3
        score += len(deg_matched)  * 2
        if has_exp and jd.min_years_experience:
            score += 2

        coverage = (
            len(req_matched) / len(jd.required_skills)
            if jd.required_skills else 0.0
        )
        return score, req_matched, nice_matched, cert_matched, deg_matched, coverage, gaps

    def _llm_summary(self, cs: CandidateScore, jd) -> str:
        """Call local Ollama for a 3-sentence candidate narrative. Silent on failure."""
        try:
            prompt = (
                f"You are an expert technical recruiter. Be concise.\n\n"
                f"Job: {jd.job_titles[0] if jd.job_titles else 'Unknown Role'}\n"
                f"Required skills: {', '.join(jd.required_skills[:8])}\n\n"
                f"Candidate matched: {', '.join(cs.required_matched)}\n"
                f"Gaps: {', '.join(cs.gaps[:5]) or 'None'}\n"
                f"Certifications: {', '.join(cs.cert_matched) or 'None'}\n"
                f"Coverage: {cs.coverage_pct:.0%}\n\n"
                f"Write exactly 3 sentences: overall fit, key strengths, main gaps."
            )
            body = json.dumps({
                "model":   self._ollama_mdl,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.2, "num_predict": 200},
            }).encode()
            req = urllib.request.Request(
                f"{self._ollama_url}/api/generate",
                data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("response", "").strip()
        except Exception as exc:
            self._log.debug("Ollama unavailable (%s) — skipping LLM summary.", exc)
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, jd, use_llm: bool = False, top_n: int = 10) -> EvaluationResult:
        """
        Main entry point.

        Parameters
        ----------
        jd      : JDProfile  (output of JobDescriptionParser.parse())
        use_llm : bool       enable Ollama narrative summaries
        top_n   : int        max candidates in output

        Returns
        -------
        EvaluationResult
        """
        from jd_parser import JDProfile  # local import to avoid circular dep
        self._log.info(
            "Evaluating: %s | required=%d, nice=%d, certs=%d",
            jd.summary(), len(jd.required_skills),
            len(jd.nice_to_have_skills), len(jd.required_certifications),
        )

        # 1. Fetch all candidate data in one query
        candidates = self._fetch_all_candidates()
        self._log.info("Scoring %d candidates…", len(candidates))

        # 2. Score every candidate
        scored: List[CandidateScore] = []
        for cid, cdata in candidates.items():
            score, req_m, nice_m, cert_m, deg_m, cov, gaps = self._score_candidate(cdata, jd)
            scored.append(CandidateScore(
                rank              = 0,          # assigned after sort
                candidate_id      = cid,
                file_name         = cdata["file_name"],
                total_score       = score,
                coverage_pct      = round(cov, 4),
                required_matched  = req_m,
                nice_matched      = nice_m,
                cert_matched      = cert_m,
                degree_matched    = deg_m,
                has_experience_node = cdata["has_exp"],
                gaps              = gaps,
            ))

        # 3. Sort: primary = total_score desc, secondary = coverage desc
        scored.sort(key=lambda c: (-c.total_score, -c.coverage_pct))
        top = scored[:top_n]

        # 4. Assign ranks + optional LLM summaries
        for i, cs in enumerate(top):
            cs.rank = i + 1
            if use_llm and cs.total_score > 0:
                cs.llm_summary = self._llm_summary(cs, jd)

        self._log.info(
            "Evaluation complete: top score=%d, coverage=%.0f%%",
            top[0].total_score if top else 0,
            (top[0].coverage_pct * 100) if top else 0,
        )
        return EvaluationResult(
            jd_source                   = jd.source_file,
            jd_job_title                = jd.job_titles[0] if jd.job_titles else "",
            jd_required_skills          = jd.required_skills,
            jd_nice_to_have             = jd.nice_to_have_skills,
            jd_certifications           = jd.required_certifications,
            jd_degrees                  = jd.required_degrees,
            jd_min_experience           = jd.min_years_experience,
            total_candidates_evaluated  = len(candidates),
            ranked_candidates           = top,
        )
