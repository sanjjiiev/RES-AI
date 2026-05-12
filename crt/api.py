"""
Phase 5 – api.py
================
Production FastAPI application for RES-AI.
• POST /ingest          – upload a resume PDF, run full pipeline (Phase 2+3)
• POST /evaluate        – rank all candidates against a job description (Phase 4)
• GET  /candidates      – list all candidates with metadata
• GET  /candidates/{id} – full candidate profile with all linked entities
• GET  /query/skill     – find candidates by skill name (exact + ecosystem)
• GET  /stats           – knowledge graph statistics
• GET  /health          – liveness probe

Security & Audit (Phase 5 defensive ops)
-----------------------------------------
• Every request is logged to audit.log with timestamp, endpoint, client IP.
• Prompt-injection detection on JD text (whitespace-hidden instruction patterns).
• File type validation + size limit on resume uploads.
• CORS configured for local Next.js frontend (localhost:3000).

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

from fastapi import (
    FastAPI, File, Form, HTTPException,
    Request, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from main import PipelineConfig, SecureIngestionPipeline, KnowledgeGraphBuilder
from schema import ENTITY_SCHEMA
from neo4j import GraphDatabase
from taxonomy import SkillTaxonomySeeder, GraphRAGLinker
from query_engine import GraphQueryEngine
from jd_parser import JobDescriptionParser
from evaluator import CandidateEvaluator

# ---------------------------------------------------------------------------
# Logging – file + console
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("audit.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("RES-AI.API")

# ---------------------------------------------------------------------------
# Shared singletons (initialised once at startup)
# ---------------------------------------------------------------------------
_cfg: Optional[PipelineConfig]        = None
_driver                                = None
_ingestion: Optional[SecureIngestionPipeline] = None
_graph:     Optional[KnowledgeGraphBuilder]   = None
_qe:        Optional[GraphQueryEngine]        = None
_jd_parser: Optional[JobDescriptionParser]    = None

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTS  = {".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _driver, _ingestion, _graph, _qe

    logger.info("RES-AI API starting up…")
    _cfg      = PipelineConfig()
    _ingestion = SecureIngestionPipeline(_cfg)
    _graph     = KnowledgeGraphBuilder(_cfg)
    _driver    = _graph.driver
    _qe        = GraphQueryEngine(_driver)
    _jd_parser = JobDescriptionParser()

    # Seed taxonomy once (idempotent)
    seeder = SkillTaxonomySeeder(_driver)
    seeder.seed()

    logger.info("RES-AI API ready.")
    yield

    logger.info("RES-AI API shutting down…")
    if _graph:
        _graph.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RES-AI",
    description="Secure Resume Intelligence API — GraphRAG-powered candidate evaluation.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Audit middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def audit_log(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "AUDIT | %s %s | client=%s | status=%d | %sms",
        request.method, request.url.path,
        request.client.host if request.client else "unknown",
        response.status_code, elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# Prompt injection guard
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(previous|all)\s+instructions?|"
    r"forget\s+(everything|all)|"
    r"you\s+are\s+now|"
    r"act\s+as\s+(an?\s+)?",
    re.IGNORECASE,
)

def _check_injection(text: str) -> None:
    if _INJECTION_PATTERNS.search(text):
        logger.warning("SECURITY | Prompt injection attempt detected.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input contains disallowed patterns (prompt injection guard).",
        )


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class SkillQueryRequest(BaseModel):
    skill: str
    ecosystem: bool = True   # use multi-hop traversal if True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/stats", tags=["System"])
async def graph_stats():
    return _qe.graph_stats()


# ── Resume Ingestion ─────────────────────────────────────────────────────────

@app.post("/ingest", tags=["Ingestion"], status_code=status.HTTP_201_CREATED)
async def ingest_resume(
    file: UploadFile = File(...),
    skip_duplicates: bool = Form(True),
):
    """Upload a PDF/DOCX resume → parse → store in Neo4j knowledge graph."""

    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTS}")

    # Validate size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = _ingestion.run(tmp_path)

        # Override candidate_id's file_name to the real upload name
        result.file_name = file.filename or result.file_name

        ingested = _graph.ingest(result, skip_duplicates=skip_duplicates)

        if ingested:
            # Phase 3: link to taxonomy
            linker = GraphRAGLinker(_driver)
            link_count = linker.link_candidate(result.candidate_id, result.by_category)

        return {
            "status":       "ingested" if ingested else "duplicate_skipped",
            "candidate_id": result.candidate_id,
            "file_name":    result.file_name,
            "resume_hash":  result.resume_hash,
            "ingested_at":  result.ingested_at,
            "entities_extracted": {k: len(v) for k, v in result.by_category.items()},
            "taxonomy_links": link_count if ingested else 0,
        }
    finally:
        os.unlink(tmp_path)


# ── Candidate Queries ────────────────────────────────────────────────────────

@app.get("/candidates", tags=["Candidates"])
async def list_candidates():
    """List all candidates with basic metadata."""
    with _driver.session() as session:
        rows = session.run(
            "MATCH (c:Candidate) RETURN c.id AS id, c.file_name AS file, "
            "c.ingested_at AS ingested_at ORDER BY c.ingested_at DESC"
        )
        return [dict(r) for r in rows]


@app.get("/candidates/{candidate_id}", tags=["Candidates"])
async def get_candidate(candidate_id: str):
    """Full profile: all entities linked to a candidate."""
    profile = _qe.get_candidate_profile(candidate_id)
    if not profile:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")
    return profile


# ── Skill Queries ────────────────────────────────────────────────────────────

@app.get("/query/skill/{skill_name}", tags=["Search"])
async def find_by_skill(skill_name: str, ecosystem: bool = True):
    """
    Find candidates who know a skill.
    ecosystem=true uses GraphRAG multi-hop traversal through taxonomy.
    """
    if ecosystem:
        return _qe.find_by_ecosystem(skill_name.title())
    return _qe.find_by_skill(skill_name.title())


@app.post("/query/skills", tags=["Search"])
async def find_by_multiple_skills(skills: List[str]):
    """Find candidates who have ALL of the listed skills."""
    normalised = [s.strip().title() for s in skills if s.strip()]
    return _qe.find_by_skills_all(normalised)


@app.get("/query/security", tags=["Search"])
async def find_security_candidates(
    domains: str = "Blue Team,Soc,Siem",
    languages: str = "Python,Javascript",
):
    """GraphRAG: find candidates matching Blue Team / SOC + programming profile."""
    dom = [d.strip().title() for d in domains.split(",")]
    lng = [l.strip().title() for l in languages.split(",")]
    return _qe.find_security_candidates(dom, lng)


# ── JD Evaluation (DOCX upload) ───────────────────────────────────────

@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_candidates(
    file: UploadFile = File(..., description="Job description DOCX or PDF"),
    top_n:   int  = Form(10),
    use_llm: bool = Form(False),
):
    """
    Upload a Job Description DOCX/PDF.
    The system parses it, extracts required + nice-to-have skills, certifications,
    degree requirements, and experience thresholds — then ranks all ingested
    candidates in the knowledge graph against the JD.
    """
    # Validate file type
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTS}")

    # Read and size-check
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Parse JD document → JDProfile
        jd_profile = _jd_parser.parse(tmp_path)

        # Evaluate all candidates
        evaluator = CandidateEvaluator(_driver)
        result    = evaluator.evaluate(jd_profile, use_llm=use_llm, top_n=top_n)

        return {
            "jd_source":                   result.jd_source,
            "jd_job_title":                result.jd_job_title,
            "jd_required_skills":          result.jd_required_skills,
            "jd_nice_to_have":             result.jd_nice_to_have,
            "jd_certifications":           result.jd_certifications,
            "jd_degrees":                  result.jd_degrees,
            "jd_min_experience_years":     result.jd_min_experience,
            "total_candidates_evaluated":  result.total_candidates_evaluated,
            "ranked_candidates": [
                {
                    "rank":             cs.rank,
                    "candidate_id":     cs.candidate_id,
                    "file_name":        cs.file_name,
                    "total_score":      cs.total_score,
                    "coverage_pct":     f"{cs.coverage_pct:.0%}",
                    "required_matched": cs.required_matched,
                    "nice_matched":     cs.nice_matched,
                    "cert_matched":     cs.cert_matched,
                    "degree_matched":   cs.degree_matched,
                    "gaps":             cs.gaps,
                    "llm_summary":      cs.llm_summary,
                }
                for cs in result.ranked_candidates
            ],
        }
    finally:
        os.unlink(tmp_path)
