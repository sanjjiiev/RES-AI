"""
Phase 4 – evaluator.py
=======================
JD-based candidate evaluation using:
• GLiNER  : parse required skills from raw job description text
• Neo4j   : graph-based candidate ranking (query_engine.py)
• Ollama  : optional LLM reasoning for final narrative summary
            (uses the local model specified in OLLAMA_MODEL env var)

DSPy-style structured signature is emulated with dataclasses since
we're running fully local (no OpenAI dependency).
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

logger = logging.getLogger("RES-AI.Evaluator")

# ---------------------------------------------------------------------------
# DSPy-style Signature (structured I/O contract)
# ---------------------------------------------------------------------------

@dataclass
class JDSignature:
    """Input contract for job description evaluation."""
    raw_jd_text: str
    top_n: int = 10
    nice_to_have: List[str] = field(default_factory=list)


@dataclass
class CandidateScore:
    """Output contract for one ranked candidate."""
    candidate_id: str
    file_name: str
    score: int
    required_matched: List[str]
    nice_matched: List[str]
    coverage_pct: float           # required skills covered
    llm_summary: Optional[str] = None


@dataclass
class EvaluationResult:
    """Full output for a JD evaluation run."""
    jd_required_skills: List[str]
    jd_nice_to_have: List[str]
    ranked_candidates: List[CandidateScore]
    total_candidates_evaluated: int


# ---------------------------------------------------------------------------
# JD PARSER (GLiNER-based)
# ---------------------------------------------------------------------------

class JobDescriptionParser:
    """
    Extracts required technical skills from raw JD text using GLiNER —
    the same model used for resume parsing, so vocabulary is consistent.
    """

    JD_LABELS = [
        "Programming Language", "Framework", "Database",
        "Cloud Platform", "Security Tool", "Certification",
        "Soft Skill", "Industry", "Job Title",
    ]

    def __init__(self, gliner_model=None) -> None:
        self._log = logging.getLogger("RES-AI.JDParser")
        if gliner_model is None:
            from gliner import GLiNER
            model_name = os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
            self._log.info("Loading GLiNER for JD parsing: %s", model_name)
            gliner_model = GLiNER.from_pretrained(model_name)
        self.model = gliner_model

    def parse(self, jd_text: str, threshold: float = 0.35) -> Dict[str, List[str]]:
        """Extract and deduplicate skill entities from a JD."""
        import re
        entities = self.model.predict_entities(jd_text, self.JD_LABELS, threshold=threshold)
        result: Dict[str, List[str]] = {}
        for ent in entities:
            norm = re.sub(r"\s+", " ", ent["text"].strip()).title()
            result.setdefault(ent["label"], [])
            if norm not in result[ent["label"]]:
                result[ent["label"]].append(norm)
        self._log.info("JD parsed: %d skill categories found.", len(result))
        return result


# ---------------------------------------------------------------------------
# CANDIDATE EVALUATOR
# ---------------------------------------------------------------------------

class CandidateEvaluator:
    """
    Scores and ranks all candidates in the knowledge graph against a JD.

    Workflow (DSPy-style)
    ─────────────────────
    1. Parse JD with GLiNER → required + nice-to-have skill lists
    2. Query Neo4j graph  → ranked candidate list with match counts
    3. (Optional) Send top candidates to local Ollama LLM for a narrative
       reasoning summary (Bootstrap-style: structured prompt → structured output)
    """

    def __init__(self, driver, gliner_model=None) -> None:
        self.driver    = driver
        self._log      = logging.getLogger("RES-AI.Evaluator")
        self.jd_parser = JobDescriptionParser(gliner_model)

        # Optional Ollama client
        self._ollama_url   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flat_skills(self, parsed: Dict[str, List[str]]) -> List[str]:
        """Flatten category dict into a single list of skill names."""
        return [skill for skills in parsed.values() for skill in skills]

    def _ollama_summarise(self, candidate_id: str, score: CandidateScore, jd_text: str) -> str:
        """
        Call local Ollama for a structured reasoning summary.
        Returns empty string on failure (LLM is optional).
        """
        try:
            import urllib.request
            prompt = (
                f"You are an expert technical recruiter.\n\n"
                f"Job Description:\n{jd_text[:800]}\n\n"
                f"Candidate ID: {candidate_id}\n"
                f"Required skills matched: {score.required_matched}\n"
                f"Nice-to-have matched: {score.nice_matched}\n"
                f"Coverage: {score.coverage_pct:.0%}\n\n"
                f"Write a 3-sentence evaluation of this candidate for this role. "
                f"Be concise and factual. Highlight gaps if coverage < 60%."
            )
            body = json.dumps({
                "model": self._ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200},
            }).encode()
            req  = urllib.request.Request(
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

    def evaluate(self, sig: JDSignature, use_llm: bool = False) -> EvaluationResult:
        """
        Main entry point.  Pass a JDSignature, get back an EvaluationResult.
        """
        from query_engine import GraphQueryEngine

        self._log.info("Evaluating JD (top_n=%d, llm=%s)…", sig.top_n, use_llm)

        # Step 1: Parse JD
        parsed = self.jd_parser.parse(sig.raw_jd_text)
        required   = self._flat_skills(parsed)
        nice       = sig.nice_to_have
        self._log.info("JD skills: %d required, %d nice-to-have.", len(required), len(nice))

        # Step 2: Graph ranking
        qe   = GraphQueryEngine(self.driver)
        rows = qe.rank_candidates_for_jd(required, nice, top_n=sig.top_n)

        scores: List[CandidateScore] = []
        for row in rows:
            req_matched  = row.get("req_matched", [])
            nice_matched = row.get("nice_matched", [])
            coverage     = len(req_matched) / len(required) if required else 0.0
            cs = CandidateScore(
                candidate_id     = row["candidate_id"],
                file_name        = row.get("file", ""),
                score            = row.get("score", 0),
                required_matched = req_matched,
                nice_matched     = nice_matched,
                coverage_pct     = round(coverage, 4),
            )
            # Step 3: Optional LLM narrative
            if use_llm:
                cs.llm_summary = self._ollama_summarise(cs.candidate_id, cs, sig.raw_jd_text)
            scores.append(cs)

        self._log.info(
            "Evaluation complete: %d candidates ranked. Top score: %d",
            len(scores), scores[0].score if scores else 0,
        )
        return EvaluationResult(
            jd_required_skills          = required,
            jd_nice_to_have             = nice,
            ranked_candidates           = scores,
            total_candidates_evaluated  = len(scores),
        )
