# RES-AI — Secure Resume Intelligence System

> **Production-grade, fully local, open-source resume screening pipeline powered by GraphRAG, GLiNER, and Neo4j.**

No cloud APIs. No data leaves your machine. Every decision is auditable.

---

## What Is RES-AI?

RES-AI is a 5-phase AI pipeline that turns raw resume PDFs into a queryable **knowledge graph** in Neo4j, then ranks candidates against a job description using **graph traversal** and an optional **local LLM** (Ollama).

It was built specifically to solve three problems with traditional resume screening:

| Problem | How RES-AI Solves It |
|---|---|
| Bias from names, gender, location | Microsoft Presidio scrubs all PII before any AI sees the data |
| Keyword-matching misses related skills | GraphRAG traverses a skill taxonomy (FastAPI → Python → Backend) |
| Black-box AI decisions | Every score is traceable to graph edges with confidence scores |

---

## Architecture — 5 Phases

```
PDF Resume
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 2 — Secure Ingestion  (main.py)                  │
│                                                         │
│  Docling ──► raw Markdown                               │
│      │                                                  │
│  GLiNER ──► zero-shot NER ──► structured entities       │
│      │           (Python, FastAPI, Blue Team …)         │
│      │                                                  │
│  Presidio ──► PII redacted Markdown                     │
│      │        (names, phones, emails, locations gone)   │
│      │                                                  │
│  Neo4j ──► Candidate node + typed relationship edges    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 3 — GraphRAG & Knowledge Graph  (taxonomy.py)    │
│                                                         │
│  ESCO Skill Taxonomy seeded into Neo4j                 │
│  (FastAPI → IS_FRAMEWORK_OF → Python)                  │
│                                                         │
│  GraphRAGLinker connects candidate entities             │
│  to taxonomy nodes via SIMILAR_TO edges                 │
│                                                         │
│  Multi-hop queries: "find Python devs" matches          │
│  FastAPI, Django, PyTorch, Flask candidates too         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 4 — JD Evaluation  (evaluator.py)                │
│                                                         │
│  GLiNER parses job description → required skill list   │
│  Neo4j graph ranks candidates by skill coverage        │
│  Optional: Ollama LLM writes narrative summaries        │
│  (DeepSeek-R1, Llama-3, or any local model)            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 5 — Enterprise API  (api.py)                     │
│                                                         │
│  FastAPI REST server with Swagger UI                   │
│  Audit logging to audit.log (every request)            │
│  Prompt injection guard on all JD inputs               │
│  File type + size validation on uploads                │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Document parsing | [Docling (IBM)](https://github.com/DS4SD/docling) | Preserves tables, columns, headers in PDFs |
| Entity extraction | [GLiNER](https://github.com/urchade/GLiNER) | Zero-shot NER — no fine-tuning needed |
| PII redaction | [Microsoft Presidio](https://microsoft.github.io/presidio/) | Production-grade, used at Microsoft scale |
| Knowledge graph | [Neo4j](https://neo4j.com/) | Graph DB with Cypher query language |
| Skill taxonomy | ESCO-inspired (built-in) | 100+ skill relationships across 10 domains |
| LLM (optional) | [Ollama](https://ollama.com/) | Fully local — DeepSeek-R1, Llama-3, etc. |
| REST API | [FastAPI](https://fastapi.tiangolo.com/) | Async, auto Swagger docs, production-ready |
| CLI | argparse (built-in) | No extra deps — `python run.py ingest ...` |

---

## File Structure

```
crt/
├── schema.py          # Shared ENTITY_SCHEMA — single source of truth
├── main.py            # Phase 2: ingestion orchestrator
├── taxonomy.py        # Phase 3: ESCO skill taxonomy + GraphRAG linker
├── query_engine.py    # Phase 3: graph traversal query engine
├── evaluator.py       # Phase 4: JD parser + candidate scorer
├── api.py             # Phase 5: FastAPI REST server + audit logging
├── run.py             # CLI entry point — all phases via subcommands
├── requirements.txt   # All Python dependencies
└── .env.example       # Environment variable template
```

---

## Prerequisites

Before installing, make sure you have:

- **Python 3.10+**
- **Neo4j 5.x** running locally (Docker is the easiest way)
- **Ollama** *(optional — only needed for LLM narrative summaries)*

### Start Neo4j with Docker

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/supersecretpassword \
  neo4j:5
```

Neo4j Browser will be at **http://localhost:7474**

---

## Installation

```powershell
# 1. Navigate to the project folder
cd "C:\Users\ssanj\OneDrive\Documents\HUB_2.0\RES-AI"

# 2. Activate your virtual environment
.\venv\Scripts\Activate

# 3. Install all dependencies
pip install -r crt\requirements.txt

# 4. Download the SpaCy English model (required by Presidio)
python -m spacy download en_core_web_lg

# 5. Set up environment variables
copy crt\.env.example crt\.env
# Open crt\.env and set your Neo4j password if different from default
```

---

## Usage

All commands are run from inside the `crt/` folder:

```powershell
cd crt
```

---

### Ingest a Single Resume (Phases 2 + 3)

```powershell
python run.py ingest --resume "..\sample_resume\Resume.pdf"
```

**What happens:**
1. PDF → layout-aware Markdown (Docling)
2. 12 entity categories extracted: Programming Language, Framework, Database, Cloud Platform, Security Tool, Soft Skill, Job Title, Degree, University, Certification, Years of Experience, Industry
3. All PII (name, phone, email, location) redacted
4. Candidate node + typed relationship edges written to Neo4j
5. Skill taxonomy seeded and candidate linked to taxonomy graph
6. Two output files saved: `<ID>_safe.md` and `<ID>_entities.json`

**Options:**

| Flag | Description |
|---|---|
| `--resume FILE` | Path to a single PDF or DOCX |
| `--folder DIR` | Process all PDFs in a folder (batch mode) |
| `--force` | Re-ingest even if the same file was already processed |
| `--no-taxonomy` | Skip Phase 3 taxonomy linking |
| `--json` | Print result as JSON |

---

### Batch Ingest a Folder

```powershell
python run.py ingest --folder "..\resumes_folder\"
```

---

### Evaluate Candidates Against a Job Description (Phase 4)

```powershell
python run.py evaluate --jd "Looking for a Python backend developer with FastAPI, PostgreSQL, Docker, and AWS experience."
```

```powershell
# From a text file
python run.py evaluate --jd-file job_description.txt --top-n 5
```

```powershell
# With Ollama LLM narrative summaries (requires Ollama running)
python run.py evaluate --jd "..." --llm
```

**Sample output:**
```
Rank  1 | CANDIDATE_A1B2C3D4E5F6  | score=8 | coverage=80%
         matched: Python, Fastapi, Postgresql, Docker
Rank  2 | CANDIDATE_G7H8I9J0K1L2  | score=5 | coverage=50%
         matched: Python, Aws
```

---

### Search by Skill — GraphRAG Mode (Phase 3)

```powershell
# Finds candidates who know Python OR anything in the Python ecosystem
python run.py search --skill Python

# Finds all cybersecurity candidates via taxonomy traversal
python run.py search --skill "Blue Team"

# Exact match only (no ecosystem traversal)
python run.py search --skill Python --no-ecosystem
```

---

### Knowledge Graph Statistics

```powershell
python run.py stats
```

```
========================================
RES-AI Knowledge Graph Stats
  Candidates      : 12
  Skill nodes     : 347
  Relationships   : 891
  Taxonomy nodes  : 108
========================================
```

---

### Seed / Refresh the Skill Taxonomy

```powershell
python run.py seed-taxonomy
```

Safe to run multiple times — all writes use `MERGE` (idempotent).

---

### Start the REST API (Phase 5)

```powershell
python run.py serve
# Dev mode with hot-reload:
python run.py serve --reload
```

Open **http://localhost:8000/docs** for the full interactive Swagger UI.

#### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/stats` | Knowledge graph statistics |
| `POST` | `/ingest` | Upload a resume (multipart/form-data) |
| `GET` | `/candidates` | List all candidates |
| `GET` | `/candidates/{id}` | Full candidate profile |
| `GET` | `/query/skill/{name}` | Find candidates by skill (GraphRAG) |
| `POST` | `/query/skills` | Find candidates matching ALL listed skills |
| `GET` | `/query/security` | Find Blue Team / SOC candidates |
| `POST` | `/evaluate` | Rank candidates against a JD |

---

## Neo4j Graph — Useful Queries

Open **http://localhost:7474** and run these in the Cypher browser:

```cypher
-- See the full graph
MATCH (c:Candidate)-[r]->(n) RETURN c, r, n

-- See one candidate's complete profile
MATCH (c:Candidate {id: 'CANDIDATE_XXXX'})-[r]->(n) RETURN c, r, n

-- Find all Python developers (ecosystem traversal)
MATCH (root:ProgrammingLanguage {name: 'Python'})
MATCH (related)-[:IS_FRAMEWORK_OF|IS_LIBRARY_OF*1..2]->(root)
WITH collect(related.name) + ['Python'] AS ecosystem
MATCH (c:Candidate)-[r]->(n) WHERE n.name IN ecosystem
RETURN DISTINCT c.id, collect(n.name) AS matched_skills

-- Rank candidates by skill count
MATCH (c:Candidate)-[r]->(n)
RETURN c.id, count(n) AS skill_count, collect(n.name)[..5] AS sample_skills
ORDER BY skill_count DESC

-- See the skill taxonomy
MATCH (child)-[r]->(parent)
WHERE child.taxonomy = true
RETURN child.name, type(r), parent.name
LIMIT 50

-- Check schema constraints
SHOW CONSTRAINTS
```

---

## Environment Variables

Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j bolt connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `supersecretpassword` | Neo4j password |
| `TARGET_RESUME` | *(path)* | Default resume for `python main.py` |
| `GLINER_MODEL` | `urchade/gliner_medium-v2.1` | GLiNER model name |
| `NER_THRESHOLD` | `0.40` | Confidence threshold for entity extraction |
| `MAX_UPLOAD_MB` | `20` | API upload size limit |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `deepseek-r1:8b` | Local LLM model for summaries |

---

## Security Features (Phase 5)

- **PII Redaction** — names, phones, emails, URLs, locations removed before graph storage
- **Dynamic Allow-list** — technical terms (e.g. "Python", "Firebase") are protected from being redacted by Presidio
- **SHA-256 Deduplication** — same resume won't be processed twice unless `--force` is used
- **Prompt Injection Guard** — API rejects JD text containing patterns like `"ignore previous instructions"`
- **Audit Logging** — every API request logged to `audit.log` with timestamp, endpoint, client IP, status code, and response time
- **File Validation** — only `.pdf` and `.docx` accepted, with configurable size limit

---

## Roadmap (from RES-AI.docx)

| Phase | Status |
|---|---|
| Phase 1 — Infrastructure (Docker, Neo4j, Ollama) | ✅ Manual setup (see Prerequisites) |
| Phase 2 — Secure Ingestion (Docling + GLiNER + Presidio + Neo4j) | ✅ Complete |
| Phase 3 — GraphRAG & Skill Taxonomy | ✅ Complete |
| Phase 4 — JD Evaluation (GLiNER + graph scoring + Ollama) | ✅ Complete |
| Phase 5 — FastAPI + Audit Logging | ✅ Complete |
| Next — Next.js Dashboard (visual skill graph) | 🔜 Planned |
| Next — ELK Stack / Splunk SIEM integration | 🔜 Planned |
| Next — DSPy BootstrapFewShot optimizer with golden examples | 🔜 Planned |

---

## License

Private / Internal Use. All data processed locally — no external API calls.
