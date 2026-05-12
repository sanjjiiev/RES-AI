"""
RES-AI  –  Production Resume Intelligence Pipeline  (Phases 2 & 3)
====================================================================
Phase 2 – Ingestion & Sanitisation
  • Docling  : layout-aware PDF/DOCX → Markdown
  • GLiNER   : zero-shot NER, sliding-window chunking, confidence threshold
  • Presidio : PII redaction with dynamic technical-term allow-list
  • Neo4j    : schema constraints, typed rels, batch UNWIND, deduplication

Phase 3 – GraphRAG & Knowledge Graph
  • taxonomy.py     : ESCO-inspired master skill hierarchy seeded into Neo4j
  • GraphRAGLinker  : candidate entities connected to taxonomy via SIMILAR_TO

Phase 4 – JD Evaluation  →  evaluator.py
Phase 5 – FastAPI REST API  →  api.py

Environment Variables
---------------------
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
TARGET_RESUME, GLINER_MODEL, NER_THRESHOLD
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Third-party imports (fail fast with a helpful message)
# ---------------------------------------------------------------------------
try:
    from docling.document_converter import DocumentConverter
    from gliner import GLiNER
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError as _err:
    raise ImportError(
        f"Missing dependency: {_err}\n"
        "Install with: pip install docling gliner presidio-analyzer "
        "presidio-anonymizer neo4j"
    ) from _err


# ===========================================================================
# LOGGING
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RES-AI")


# ===========================================================================
# CONFIGURATION
# ===========================================================================
@dataclass
class PipelineConfig:
    """All tuneable parameters.  Values are read from env-vars with sane defaults."""

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "supersecretpassword"))
    neo4j_pool_size: int = 10
    neo4j_timeout: int = 10

    # GLiNER
    gliner_model: str = field(default_factory=lambda: os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1"))
    ner_threshold: float = field(default_factory=lambda: float(os.getenv("NER_THRESHOLD", "0.40")))

    # Chunking (GLiNER has a ~512-token context window)
    chunk_words: int = 450      # words per chunk
    chunk_overlap: int = 50     # overlapping words between chunks

    # Entity labels for GLiNER
    entity_labels: List[str] = field(default_factory=lambda: [
        "Programming Language",
        "Framework",
        "Database",
        "Cloud Platform",
        "Security Tool",
        "Soft Skill",
        "Job Title",
        "Degree",
        "University",
        "Certification",
        "Years of Experience",
        "Industry",
    ])

    # Presidio PII types to redact
    pii_entities: List[str] = field(default_factory=lambda: [
        "PERSON",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "LOCATION",
        "URL",
        "DATE_TIME",
        "NRP",          # Nationality, Religious or Political group
    ])


# ===========================================================================
# NEO4J SCHEMA  –  imported from schema.py (shared across all modules)
# ===========================================================================
from schema import ENTITY_SCHEMA  # noqa: E402  (after warnings.filterwarnings)


# ===========================================================================
# DATA CLASSES
# ===========================================================================
@dataclass
class ExtractedEntity:
    label: str
    text: str           # normalised display text
    confidence: float


@dataclass
class ParseResult:
    candidate_id: str
    resume_hash: str
    file_name: str
    ingested_at: str            # ISO-8601 UTC
    safe_markdown: str          # PII-scrubbed markdown
    entities: List[ExtractedEntity]
    by_category: Dict[str, List[Dict]]   # {"Programming Language": [{"text":..,"confidence":..}]}


# ===========================================================================
# PHASE 2-A  :  SECURE INGESTION PIPELINE
# ===========================================================================
class SecureIngestionPipeline:
    """
    Parse → Extract Entities → Redact PII

    Key production features
    -----------------------
    • Sliding-window chunking keeps all text within GLiNER's token limit.
    • Deduplication across chunks: keeps the highest-confidence score.
    • Dynamic allow-list prevents Presidio from redacting technical terms.
    • SHA-256 hash enables downstream deduplication without storing PII.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self._log = logging.getLogger("RES-AI.Ingestion")

        self._log.info("Loading Docling (PDF → Markdown)…")
        self.converter = DocumentConverter()

        self._log.info("Loading GLiNER model: %s", config.gliner_model)
        self.gliner = GLiNER.from_pretrained(config.gliner_model)

        self._log.info("Loading Microsoft Presidio engines…")
        self.analyzer  = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        self._log.info("All ingestion engines ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(path: str) -> str:
        """Return the SHA-256 hex digest of a file for deduplication."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65_536), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse whitespace and apply title-case for consistent storage."""
        return re.sub(r"\s+", " ", text.strip()).title()

    def _sliding_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping word-windows.
        This guarantees every entity is seen in full context even near
        chunk boundaries.
        """
        words  = text.split()
        step   = self.cfg.chunk_words - self.cfg.chunk_overlap
        chunks = []
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i : i + self.cfg.chunk_words]))
            if i + self.cfg.chunk_words >= len(words):
                break
        return chunks

    # ------------------------------------------------------------------
    # Step 1 — Parse
    # ------------------------------------------------------------------

    def _parse(self, file_path: str) -> str:
        self._log.info("Parsing document: %s", file_path)
        try:
            md = self.converter.convert(file_path).document.export_to_markdown()
            self._log.info("Document parsed — %d characters extracted.", len(md))
            return md
        except Exception as exc:
            raise RuntimeError(f"Docling failed to parse '{file_path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Step 2 — Entity Extraction
    # ------------------------------------------------------------------

    def _extract(self, markdown: str) -> List[ExtractedEntity]:
        chunks = self._sliding_chunks(markdown)
        self._log.info(
            "Running GLiNER over %d chunk(s) (threshold=%.2f)…",
            len(chunks), self.cfg.ner_threshold,
        )

        # (label, normalized_text) → best confidence score across all chunks
        best: Dict[Tuple[str, str], float] = {}

        for chunk in chunks:
            predictions = self.gliner.predict_entities(
                chunk,
                self.cfg.entity_labels,
                threshold=self.cfg.ner_threshold,
            )
            for pred in predictions:
                norm  = self._normalize(pred["text"])
                key   = (pred["label"], norm)
                score = float(pred.get("score", 1.0))
                if key not in best or score > best[key]:
                    best[key] = score

        entities = [
            ExtractedEntity(label=lbl, text=txt, confidence=round(score, 4))
            for (lbl, txt), score in sorted(best.items(), key=lambda x: -x[1][1] if False else x[0])
        ]

        self._log.info(
            "Extracted %d unique entities across %d categories.",
            len(entities), len({e.label for e in entities}),
        )
        return entities

    # ------------------------------------------------------------------
    # Step 3 — PII Redaction
    # ------------------------------------------------------------------

    def _redact(self, markdown: str, allow_list: List[str]) -> str:
        self._log.info(
            "Redacting PII — %d technical terms on allow-list…", len(allow_list)
        )
        try:
            hits   = self.analyzer.analyze(
                text=markdown,
                entities=self.cfg.pii_entities,
                language="en",
                allow_list=allow_list,
            )
            result = self.anonymizer.anonymize(text=markdown, analyzer_results=hits)
            self._log.info("PII redaction done — %d entities masked.", len(hits))
            return result.text
        except Exception as exc:
            self._log.error("Presidio redaction failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, file_path: str) -> ParseResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found: {file_path}")

        candidate_id = f"CANDIDATE_{uuid.uuid4().hex[:12].upper()}"
        resume_hash  = self._sha256(file_path)
        file_name    = os.path.basename(file_path)
        ingested_at  = datetime.now(timezone.utc).isoformat()

        # Steps 1-3
        raw_md   = self._parse(file_path)
        entities = self._extract(raw_md)

        allow_list  = list({e.text for e in entities})
        safe_md     = self._redact(raw_md, allow_list)

        # Group by category for easy downstream use
        by_category: Dict[str, List[Dict]] = {}
        for ent in entities:
            by_category.setdefault(ent.label, []).append(
                {"text": ent.text, "confidence": ent.confidence}
            )

        return ParseResult(
            candidate_id  = candidate_id,
            resume_hash   = resume_hash,
            file_name     = file_name,
            ingested_at   = ingested_at,
            safe_markdown = safe_md,
            entities      = entities,
            by_category   = by_category,
        )


# ===========================================================================
# PHASE 2-B  :  NEO4J KNOWLEDGE GRAPH BUILDER
# ===========================================================================
class KnowledgeGraphBuilder:
    """
    Writes a ParseResult into Neo4j with production-grade patterns:

    • Schema constraints enforced on every startup (idempotent).
    • Typed relationships per entity category (not a single generic HAS_SKILL).
    • UNWIND-based batch writes — one transaction per category instead of
      one transaction per entity (N+1 eliminated).
    • Relationship properties: confidence score + ingested_at timestamp.
    • Candidate node: id, resume_hash, file_name, ingested_at, updated_at.
    • Duplicate detection via SHA-256 resume hash — safe to re-run.
    """

    SCHEMA = ENTITY_SCHEMA

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg  = config
        self._log = logging.getLogger("RES-AI.GraphBuilder")

        self._log.info("Connecting to Neo4j at %s …", config.neo4j_uri)
        try:
            self.driver = GraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_user, config.neo4j_password),
                max_connection_pool_size=config.neo4j_pool_size,
                connection_timeout=config.neo4j_timeout,
            )
            self.driver.verify_connectivity()
            self._log.info("Neo4j connection verified.")
        except ServiceUnavailable as exc:
            raise ConnectionError(
                f"Neo4j not reachable at '{config.neo4j_uri}'. "
                "Ensure the database is running."
            ) from exc
        except AuthError as exc:
            raise PermissionError(
                "Neo4j authentication failed — check NEO4J_USER / NEO4J_PASSWORD."
            ) from exc

        self._bootstrap_schema()

    # ------------------------------------------------------------------
    # Schema bootstrap (idempotent — safe to call on every run)
    # ------------------------------------------------------------------

    def _bootstrap_schema(self) -> None:
        self._log.info("Bootstrapping Neo4j schema constraints…")
        stmts: List[str] = [
            # Candidate uniqueness
            "CREATE CONSTRAINT candidate_id_unique IF NOT EXISTS "
            "FOR (c:Candidate) REQUIRE c.id IS UNIQUE",

            "CREATE CONSTRAINT candidate_hash_unique IF NOT EXISTS "
            "FOR (c:Candidate) REQUIRE c.resume_hash IS UNIQUE",
        ]
        # Entity node uniqueness (one constraint per node label)
        for node_label, _ in self.SCHEMA.values():
            stmts.append(
                f"CREATE CONSTRAINT {node_label.lower()}_name_unique IF NOT EXISTS "
                f"FOR (n:{node_label}) REQUIRE n.name IS UNIQUE"
            )

        with self.driver.session() as session:
            for stmt in stmts:
                try:
                    session.run(stmt)
                except Exception as exc:
                    # Constraints may already exist — log and continue
                    self._log.debug("Schema note: %s", exc)

        self._log.info("Schema ready.")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def _upsert_candidate(self, session, result: ParseResult) -> None:
        """MERGE Candidate node; update metadata on re-ingestion."""
        session.run(
            """
            MERGE (c:Candidate {id: $id})
            ON CREATE SET
                c.resume_hash = $hash,
                c.file_name   = $file_name,
                c.ingested_at = $ingested_at,
                c.updated_at  = $ingested_at
            ON MATCH SET
                c.resume_hash = $hash,
                c.file_name   = $file_name,
                c.updated_at  = $ingested_at
            """,
            id          = result.candidate_id,
            hash        = result.resume_hash,
            file_name   = result.file_name,
            ingested_at = result.ingested_at,
        )

    def _batch_upsert(
        self,
        session,
        candidate_id: str,
        node_label: str,
        rel_type: str,
        items: List[Dict],
        ingested_at: str,
    ) -> None:
        """
        UNWIND batch write: one Cypher round-trip per category.
        Merges the entity node, then merges the relationship with metadata.
        """
        if not items:
            return
        # f-string for labels/rel-types is safe here because all values come
        # from the hard-coded ENTITY_SCHEMA dict — not from user input.
        session.run(
            f"""
            MATCH (c:Candidate {{id: $candidate_id}})
            UNWIND $items AS item
            MERGE (n:{node_label} {{name: item.text}})
              ON CREATE SET n.created_at = $ingested_at
            MERGE (c)-[r:{rel_type}]->(n)
              ON CREATE SET r.confidence = item.confidence, r.ingested_at = $ingested_at
              ON MATCH  SET r.confidence = item.confidence, r.ingested_at = $ingested_at
            """,
            candidate_id = candidate_id,
            items        = items,
            ingested_at  = ingested_at,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(self, resume_hash: str) -> bool:
        """Returns True if a resume with this SHA-256 hash already exists."""
        with self.driver.session() as session:
            rec = session.run(
                "MATCH (c:Candidate {resume_hash: $h}) RETURN c.id AS id LIMIT 1",
                h=resume_hash,
            ).single()
            return rec is not None

    def ingest(self, result: ParseResult, skip_duplicates: bool = True) -> bool:
        """
        Persist a ParseResult to Neo4j.
        Returns True if ingested, False if skipped as a duplicate.
        """
        if skip_duplicates and self.is_duplicate(result.resume_hash):
            self._log.warning(
                "Duplicate resume detected (hash prefix: %s…) — skipping.",
                result.resume_hash[:16],
            )
            return False

        self._log.info("Writing graph for candidate %s …", result.candidate_id)
        total = 0

        with self.driver.session() as session:
            self._upsert_candidate(session, result)

            for gliner_label, items in result.by_category.items():
                if gliner_label not in self.SCHEMA:
                    self._log.debug("No schema mapping for label '%s' — skipped.", gliner_label)
                    continue

                node_label, rel_type = self.SCHEMA[gliner_label]
                self._batch_upsert(
                    session, result.candidate_id,
                    node_label, rel_type, items, result.ingested_at,
                )
                self._log.debug(
                    "  ✓ [%s] → %d %s node(s)", rel_type, len(items), node_label
                )
                total += len(items)

        self._log.info(
            "Graph write complete: %d entity nodes linked to %s.",
            total, result.candidate_id,
        )
        return True

    def close(self) -> None:
        self.driver.close()
        self._log.info("Neo4j driver closed.")


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================
def run_pipeline(
    file_path: str,
    config: Optional[PipelineConfig] = None,
    skip_duplicates: bool = True,
    run_phase3: bool = True,
) -> Optional[ParseResult]:
    """
    End-to-end Phases 2 & 3 orchestrator.

    Phase 2: parse → extract → redact → store in Neo4j
    Phase 3: seed taxonomy + link candidate to master skill graph

    Returns ParseResult on success, None on failure.
    Saves artefacts: <ID>_safe.md and <ID>_entities.json
    """
    if config is None:
        config = PipelineConfig()

    logger.info("=" * 62)
    logger.info("  RES-AI  |  Phase 2 + 3 Pipeline")
    logger.info("  Target : %s", file_path)
    logger.info("=" * 62)

    ingestion = SecureIngestionPipeline(config)
    graph     = KnowledgeGraphBuilder(config)

    try:
        # ── Step 1: Parse, extract, redact ───────────────────────────
        result = ingestion.run(file_path)

        logger.info("-" * 62)
        logger.info("EXTRACTION SUMMARY  (candidate: %s)", result.candidate_id)
        logger.info("  Resume hash : %s…", result.resume_hash[:24])
        for cat, items in sorted(result.by_category.items()):
            sample = ", ".join(i["text"] for i in items[:4])
            more   = f" (+{len(items)-4} more)" if len(items) > 4 else ""
            logger.info("  %-24s → %s%s", cat, sample, more)
        logger.info("-" * 62)

        # ── Step 2: Persist to Neo4j ─────────────────────────────────
        ingested = graph.ingest(result, skip_duplicates=skip_duplicates)

        if ingested:
            # ── Step 3: Seed taxonomy + link candidate ────────────────
            if run_phase3:
                try:
                    from taxonomy import SkillTaxonomySeeder, GraphRAGLinker
                    logger.info("-" * 62)
                    logger.info("PHASE 3 | Skill taxonomy & GraphRAG linking")
                    SkillTaxonomySeeder(graph.driver).seed()
                    link_count = GraphRAGLinker(graph.driver).link_candidate(
                        result.candidate_id, result.by_category
                    )
                    logger.info("PHASE 3 | %d taxonomy edges created.", link_count)
                except ImportError:
                    logger.warning("taxonomy.py not found — skipping Phase 3.")

            out_dir = os.path.dirname(os.path.abspath(file_path))

            # Save PII-safe Markdown
            md_path = os.path.join(out_dir, f"{result.candidate_id}_safe.md")
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(f"# {result.candidate_id}\n")
                fh.write(f"# Ingested: {result.ingested_at}\n\n")
                fh.write(result.safe_markdown)
            logger.info("Safe Markdown → %s", md_path)

            # Save structured JSON
            json_path = os.path.join(out_dir, f"{result.candidate_id}_entities.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "candidate_id": result.candidate_id,
                        "resume_hash":  result.resume_hash,
                        "file_name":    result.file_name,
                        "ingested_at":  result.ingested_at,
                        "entities":     result.by_category,
                    },
                    fh, indent=2, ensure_ascii=False,
                )
            logger.info("Entity JSON  → %s", json_path)

            logger.info("=" * 62)
            logger.info("  ✅  PIPELINE COMPLETE")
            logger.info("  Neo4j Browser : http://localhost:7474")
            logger.info(
                "  Cypher        : MATCH (c:Candidate {id: '%s'})-[r]->(n) RETURN c,r,n",
                result.candidate_id,
            )
            logger.info("=" * 62)

        return result

    except FileNotFoundError as exc:
        logger.error("File not found — %s", exc)
    except (ConnectionError, PermissionError) as exc:
        logger.error("Database error — %s", exc)
    except RuntimeError as exc:
        logger.error("Pipeline runtime error — %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure — %s", exc)
    finally:
        graph.close()

    return None


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    _config = PipelineConfig()   # reads NEO4J_* and GLINER_MODEL from env

    _target = os.getenv(
        "TARGET_RESUME",
        r"C:\Users\ssanj\OneDrive\Documents\HUB_2.0\RES-AI\sample_resume\Resume.pdf",
    )

    run_pipeline(_target, _config, skip_duplicates=True)