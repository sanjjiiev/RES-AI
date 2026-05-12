"""
run.py – RES-AI Command-Line Interface
========================================
Ties together all phases into a single entry point.

Usage examples
--------------
# Ingest one resume (Phases 2 + 3)
python run.py ingest --resume path/to/Resume.pdf

# Ingest all PDFs in a folder
python run.py ingest --folder path/to/resumes/

# Evaluate candidates against a job description (Phase 4)
python run.py evaluate --jd "We need a Python dev with FastAPI and PostgreSQL..."

# Evaluate with Ollama LLM summaries enabled
python run.py evaluate --jd "..." --llm

# Search for candidates by skill (GraphRAG ecosystem traversal)
python run.py search --skill Python

# Show graph stats
python run.py stats

# Seed / refresh the skill taxonomy (Phase 3)
python run.py seed-taxonomy

# Start the FastAPI REST API (Phase 5)
python run.py serve
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RES-AI.CLI")

ALLOWED_EXTS = {".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_config():
    from main import PipelineConfig
    return PipelineConfig()


def _get_driver(cfg):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
    driver.verify_connectivity()
    return driver


def _print_json(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_ingest(args):
    from main import run_pipeline, PipelineConfig

    cfg = _get_config()

    targets: list[Path] = []
    if args.resume:
        targets = [Path(args.resume)]
    elif args.folder:
        folder = Path(args.folder)
        targets = [f for f in folder.iterdir() if f.suffix.lower() in ALLOWED_EXTS]
        logger.info("Found %d resume(s) in %s", len(targets), folder)

    if not targets:
        logger.error("No resumes found. Use --resume FILE or --folder DIR.")
        sys.exit(1)

    results = []
    for path in targets:
        logger.info("─" * 50)
        result = run_pipeline(
            str(path), cfg,
            skip_duplicates=not args.force,
            run_phase3=not args.no_taxonomy,
        )
        if result:
            results.append({
                "candidate_id": result.candidate_id,
                "file": result.file_name,
                "entities": {k: len(v) for k, v in result.by_category.items()},
            })

    logger.info("=" * 50)
    logger.info("Ingested %d / %d resume(s).", len(results), len(targets))
    if args.json:
        _print_json(results)


def cmd_evaluate(args):
    from evaluator import CandidateEvaluator, JDSignature

    cfg    = _get_config()
    driver = _get_driver(cfg)

    jd_text = args.jd
    if not jd_text and args.jd_file:
        jd_text = Path(args.jd_file).read_text(encoding="utf-8")

    if not jd_text:
        logger.error("Provide --jd 'text' or --jd-file path.")
        sys.exit(1)

    evaluator = CandidateEvaluator(driver)
    sig       = JDSignature(raw_jd_text=jd_text, top_n=args.top_n)
    result    = evaluator.evaluate(sig, use_llm=args.llm)

    driver.close()

    logger.info("\n%s", "=" * 60)
    logger.info("JD EVALUATION RESULTS")
    logger.info("Required skills: %s", ", ".join(result.jd_required_skills[:10]))
    logger.info("%s", "=" * 60)

    for i, cs in enumerate(result.ranked_candidates):
        logger.info(
            "Rank %2d | %-40s | score=%d | coverage=%s",
            i + 1, cs.candidate_id, cs.score, f"{cs.coverage_pct:.0%}",
        )
        if cs.required_matched:
            logger.info("         matched: %s", ", ".join(cs.required_matched[:6]))
        if cs.llm_summary:
            logger.info("         summary: %s", cs.llm_summary)

    if args.json:
        _print_json([
            {
                "rank": i + 1,
                "candidate_id": cs.candidate_id,
                "score": cs.score,
                "coverage": f"{cs.coverage_pct:.0%}",
                "required_matched": cs.required_matched,
                "nice_matched": cs.nice_matched,
                "llm_summary": cs.llm_summary,
            }
            for i, cs in enumerate(result.ranked_candidates)
        ])


def cmd_search(args):
    from query_engine import GraphQueryEngine

    cfg    = _get_config()
    driver = _get_driver(cfg)
    qe     = GraphQueryEngine(driver)

    skill = args.skill.strip().title()
    if args.ecosystem:
        results = qe.find_by_ecosystem(skill, max_hops=args.hops)
    else:
        results = qe.find_by_skill(skill)

    driver.close()

    logger.info("Found %d candidate(s) for skill: %s", len(results), skill)
    _print_json(results)


def cmd_stats(args):
    from query_engine import GraphQueryEngine

    cfg    = _get_config()
    driver = _get_driver(cfg)
    stats  = GraphQueryEngine(driver).graph_stats()
    driver.close()

    logger.info("\n%s", "=" * 40)
    logger.info("RES-AI Knowledge Graph Stats")
    logger.info("  Candidates      : %d", stats["candidates"])
    logger.info("  Skill nodes     : %d", stats["skill_nodes"])
    logger.info("  Relationships   : %d", stats["relationships"])
    logger.info("  Taxonomy nodes  : %d", stats["taxonomy_nodes"])
    logger.info("%s", "=" * 40)


def cmd_seed_taxonomy(args):
    from taxonomy import SkillTaxonomySeeder

    cfg    = _get_config()
    driver = _get_driver(cfg)
    count  = SkillTaxonomySeeder(driver).seed()
    driver.close()
    logger.info("Taxonomy seeded: %d relationships.", count)


def cmd_serve(args):
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="RES-AI  –  Production Resume Intelligence CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ──────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest resume(s) into the knowledge graph")
    grp = p_ingest.add_mutually_exclusive_group(required=True)
    grp.add_argument("--resume", metavar="FILE",   help="Path to a single PDF/DOCX")
    grp.add_argument("--folder", metavar="DIR",    help="Folder containing resumes")
    p_ingest.add_argument("--force",        action="store_true", help="Re-ingest even if duplicate")
    p_ingest.add_argument("--no-taxonomy",  action="store_true", help="Skip Phase 3 taxonomy linking")
    p_ingest.add_argument("--json",         action="store_true", help="Print result as JSON")
    p_ingest.set_defaults(func=cmd_ingest)

    # ── evaluate ─────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Rank candidates against a job description")
    jd_grp = p_eval.add_mutually_exclusive_group(required=True)
    jd_grp.add_argument("--jd",      metavar="TEXT", help="Job description text (quoted)")
    jd_grp.add_argument("--jd-file", metavar="FILE", help="Path to JD text file")
    p_eval.add_argument("--top-n",   type=int, default=10, help="Max candidates to return (default 10)")
    p_eval.add_argument("--llm",     action="store_true",  help="Use Ollama LLM for narrative summaries")
    p_eval.add_argument("--json",    action="store_true",  help="Print result as JSON")
    p_eval.set_defaults(func=cmd_evaluate)

    # ── search ───────────────────────────────────────────────────────────
    p_search = sub.add_parser("search", help="Find candidates by skill name")
    p_search.add_argument("--skill",     required=True, metavar="NAME")
    p_search.add_argument("--ecosystem", action="store_true", default=True,
                           help="Use GraphRAG multi-hop traversal (default: true)")
    p_search.add_argument("--hops",      type=int, default=2, help="Max taxonomy hops (default 2)")
    p_search.set_defaults(func=cmd_search)

    # ── stats ─────────────────────────────────────────────────────────────
    p_stats = sub.add_parser("stats", help="Show knowledge graph statistics")
    p_stats.set_defaults(func=cmd_stats)

    # ── seed-taxonomy ─────────────────────────────────────────────────────
    p_seed = sub.add_parser("seed-taxonomy", help="Seed/refresh the master skill taxonomy in Neo4j")
    p_seed.set_defaults(func=cmd_seed_taxonomy)

    # ── serve ─────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start the FastAPI REST API")
    p_serve.add_argument("--host",   default="0.0.0.0")
    p_serve.add_argument("--port",   type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Enable hot-reload (dev mode)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
