"""
jd_parser.py – Job Description Document Parser
================================================
Converts a DOCX/PDF job description into a structured JDProfile.

Strategy
--------
1. Docling converts the DOCX to layout-aware Markdown (preserves headings,
   bullet points, tables — all common in JD documents).
2. GLiNER extracts technical entities (same model used for resumes, so the
   vocabulary is consistent and skill matching works correctly).
3. Section-based heuristic splits entities into "required" vs "nice-to-have"
   by detecting common heading patterns:
       Required / Must Have / Mandatory  →  required
       Preferred / Nice to Have / Good to Have / Bonus  →  nice_to_have
4. A keyword scanner extracts years-of-experience ranges, degree requirements,
   and key responsibilities from the raw text.

Output: JDProfile dataclass — consumed by evaluator.py for candidate ranking.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

logger = logging.getLogger("RES-AI.JDParser")

# ---------------------------------------------------------------------------
# Section heading patterns
# ---------------------------------------------------------------------------
_REQUIRED_HEADINGS = re.compile(
    r"(must[\s\-]have|required|mandatory|essential|minimum qualifications?"
    r"|what you need|you (must|will need)|key requirements?)",
    re.IGNORECASE,
)
_NICE_HEADINGS = re.compile(
    r"(nice[\s\-]to[\s\-]have|preferred|bonus|good[\s\-]to[\s\-]have"
    r"|advantageous|desirable|plus|optional|what.*bonus)",
    re.IGNORECASE,
)
_RESPONSIBILITY_HEADINGS = re.compile(
    r"(responsibilities|what you.ll do|your role|duties|day[\s\-]to[\s\-]day"
    r"|about the role|the job|what we expect)",
    re.IGNORECASE,
)

# Experience patterns: "3+ years", "5-7 years", "minimum 2 years"
_EXP_PATTERN = re.compile(
    r"(\d+)\s*[\+\-]?\s*(?:to\s*\d+)?\s*(?:\+)?\s*years?",
    re.IGNORECASE,
)

# Degree patterns
_DEGREE_PATTERN = re.compile(
    r"\b(b\.?tech|b\.?e|b\.?sc|bachelor|master|m\.?tech|m\.?sc|phd|mba"
    r"|undergraduate|postgraduate|diploma)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class JDProfile:
    """
    Fully structured representation of a parsed job description.
    Consumed by CandidateEvaluator in evaluator.py.
    """
    source_file: str                          # Original DOCX/PDF filename
    raw_markdown: str                         # Full Docling-parsed text

    # Extracted metadata
    job_titles: List[str]     = field(default_factory=list)
    company_info: str         = ""
    industry: List[str]       = field(default_factory=list)

    # Skill requirements (split by section)
    required_skills: List[str]      = field(default_factory=list)  # must-have
    nice_to_have_skills: List[str]  = field(default_factory=list)  # preferred
    all_skills: List[str]           = field(default_factory=list)  # union

    # Education & experience
    required_degrees: List[str]         = field(default_factory=list)
    required_certifications: List[str]  = field(default_factory=list)
    min_years_experience: Optional[int] = None

    # Responsibilities (for LLM context)
    responsibilities: List[str] = field(default_factory=list)

    # Raw entity categories (for debug / display)
    entities_by_category: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable one-liner for logging."""
        return (
            f"JD: {self.job_titles[0] if self.job_titles else 'Unknown'} | "
            f"{len(self.required_skills)} required skills | "
            f"{len(self.nice_to_have_skills)} nice-to-have | "
            f"exp≥{self.min_years_experience}yr"
        )


# ---------------------------------------------------------------------------
# JD PARSER
# ---------------------------------------------------------------------------
class JobDescriptionParser:
    """
    Parses a DOCX/PDF job description into a JDProfile.

    Reuses Docling for document conversion and GLiNER for entity extraction
    — same models as the resume pipeline so skill names normalise identically.
    """

    # GLiNER labels — broader than resume labels to capture JD-specific info
    JD_LABELS = [
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
        "Company",
        "Responsibility",
    ]

    def __init__(self, gliner_model=None) -> None:
        self._log = logging.getLogger("RES-AI.JDParser")

        self._log.info("Loading Docling for JD parsing…")
        from docling.document_converter import DocumentConverter
        self._converter = DocumentConverter()

        if gliner_model is None:
            from gliner import GLiNER
            model_name = os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
            self._log.info("Loading GLiNER: %s", model_name)
            gliner_model = GLiNER.from_pretrained(model_name)
        self._gliner = gliner_model

        self._threshold = float(os.getenv("NER_THRESHOLD", "0.35"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).title()

    def _parse_document(self, file_path: str) -> str:
        """Docling: DOCX/PDF → layout-aware Markdown."""
        self._log.info("Parsing JD document: %s", file_path)
        try:
            md = self._converter.convert(file_path).document.export_to_markdown()
            self._log.info("JD parsed — %d characters.", len(md))
            return md
        except Exception as exc:
            raise RuntimeError(f"Docling failed on JD '{file_path}': {exc}") from exc

    def _extract_entities(self, markdown: str) -> Dict[str, List[str]]:
        """GLiNER NER over the full JD markdown."""
        entities = self._gliner.predict_entities(
            markdown, self.JD_LABELS, threshold=self._threshold
        )
        result: Dict[str, List[str]] = {}
        seen = set()
        for ent in entities:
            norm = self._normalize(ent["text"])
            key  = (ent["label"], norm)
            if key not in seen:
                result.setdefault(ent["label"], []).append(norm)
                seen.add(key)
        return result

    def _split_by_section(self, markdown: str, entities: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """
        Heuristic section splitter:
        Assigns each skill to 'required' or 'nice_to_have' based on
        which section heading appears before it in the document.

        Falls back to putting all skills in 'required' if no sections found.
        """
        lines = markdown.split("\n")

        # Build a map: line_index → section_type
        section_map: Dict[int, str] = {}
        current = "required"   # default assumption
        for i, line in enumerate(lines):
            if _REQUIRED_HEADINGS.search(line):
                current = "required"
                section_map[i] = "required"
            elif _NICE_HEADINGS.search(line):
                current = "nice_to_have"
                section_map[i] = "nice_to_have"
            elif _RESPONSIBILITY_HEADINGS.search(line):
                current = "responsibility"
                section_map[i] = "responsibility"

        # For each extracted skill, find which section it first appears in
        skill_labels = [
            "Programming Language", "Framework", "Database",
            "Cloud Platform", "Security Tool", "Certification",
        ]
        all_skills = []
        for label in skill_labels:
            all_skills.extend(entities.get(label, []))

        required: List[str] = []
        nice: List[str]     = []
        assigned            = set()

        for skill in all_skills:
            if skill in assigned:
                continue
            # Find first line containing this skill (case-insensitive)
            skill_lower = skill.lower()
            found_section = "required"
            for i, line in enumerate(lines):
                if skill_lower in line.lower():
                    # Find the most recent section heading before this line
                    active = "required"
                    for si in sorted(section_map.keys()):
                        if si <= i:
                            active = section_map[si]
                    found_section = active
                    break

            if found_section == "nice_to_have":
                nice.append(skill)
            else:
                required.append(skill)
            assigned.add(skill)

        # If nothing was classified as nice-to-have, split by confidence heuristic:
        # if all ended up in required but there were nice-to-have headings found,
        # leave as-is. The evaluator handles both lists gracefully.
        return required, nice

    def _extract_responsibilities(self, markdown: str) -> List[str]:
        """Extract bullet points from responsibility sections."""
        lines = markdown.split("\n")
        in_section = False
        bullets: List[str] = []
        for line in lines:
            if _RESPONSIBILITY_HEADINGS.search(line):
                in_section = True
                continue
            if in_section:
                # Stop at next heading
                if re.match(r"^#+\s", line) or _REQUIRED_HEADINGS.search(line) or _NICE_HEADINGS.search(line):
                    in_section = False
                    continue
                # Collect bullet points
                clean = re.sub(r"^[\*\-\•]\s*", "", line).strip()
                if len(clean) > 15:
                    bullets.append(clean)
        return bullets[:20]  # cap at 20 responsibilities

    def _extract_min_experience(self, markdown: str) -> Optional[int]:
        """Find the minimum years of experience mentioned in the JD."""
        matches = _EXP_PATTERN.findall(markdown)
        if not matches:
            return None
        years = [int(m) for m in matches if int(m) <= 20]  # ignore noise like "2024"
        return min(years) if years else None

    def _extract_degrees(self, markdown: str, entities: Dict[str, List[str]]) -> List[str]:
        """Merge GLiNER degree entities with regex fallback."""
        gliner_degrees = entities.get("Degree", [])
        regex_degrees  = [self._normalize(m) for m in _DEGREE_PATTERN.findall(markdown)]
        combined = list(dict.fromkeys(gliner_degrees + regex_degrees))  # deduplicate, preserve order
        return combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str) -> JDProfile:
        """
        Parse a JD DOCX/PDF and return a fully structured JDProfile.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JD file not found: {file_path}")

        source_file = os.path.basename(file_path)
        markdown    = self._parse_document(file_path)
        entities    = self._extract_entities(markdown)

        required, nice = self._split_by_section(markdown, entities)

        # Build flat all_skills (union, deduplicated)
        all_skills = list(dict.fromkeys(required + nice))

        profile = JDProfile(
            source_file       = source_file,
            raw_markdown      = markdown,
            job_titles        = entities.get("Job Title", []),
            company_info      = ", ".join(entities.get("Company", [])),
            industry          = entities.get("Industry", []),
            required_skills   = required,
            nice_to_have_skills = nice,
            all_skills        = all_skills,
            required_degrees        = self._extract_degrees(markdown, entities),
            required_certifications = entities.get("Certification", []),
            min_years_experience    = self._extract_min_experience(markdown),
            responsibilities        = self._extract_responsibilities(markdown),
            entities_by_category    = {k: v for k, v in entities.items()},
        )

        self._log.info("JD profile built: %s", profile.summary())
        return profile
