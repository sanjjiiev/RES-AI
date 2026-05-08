"""
Dependencies required:
pip install docling gliner presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
"""

import os
import json
import warnings
from typing import Tuple, Dict, List

# Suppress noisy warnings from transformers/PyTorch for a cleaner terminal
warnings.filterwarnings("ignore")

from docling.document_converter import DocumentConverter
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from gliner import GLiNER

class SecureResumePipeline:
    def __init__(self):
        print("[*] Initializing Pipeline Engines...")
        
        # 1. Document Parsing Engine
        print(" ├── Loading Docling (PDF -> Markdown)")
        self.converter = DocumentConverter()
        
        # 2. NLP Extraction Engine
        print(" ├── Loading GLiNER (Zero-Shot Entity Extraction)")
        # Using medium-v2.1 for a great balance of speed and accuracy
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        
        # The specific taxonomy we want to extract for our Graph DB
        self.tech_labels = [
            "Skill", "Programming Language", "Framework", "Database", 
            "Cloud Platform", "Security Tool", "Job Title", "Degree", "University"
        ]
        
        # 3. PII Redaction Engine
        print(" └── Loading Presidio (PII Shield)")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # We target specific demographics that cause bias or data leaks
        self.pii_entities = [
            "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", 
            "US_SSN", "UK_NHS" # Add more regional PII types if needed
        ]
        print("[*] All engines loaded and ready.\n")

    def run(self, file_path: str) -> Tuple[str, Dict]:
        """
        Executes the full pipeline: Parse -> Extract Entities -> Redact PII.
        Returns the safe markdown text and the structured JSON data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found at path: {file_path}")

        # STEP 1: Parse PDF to Layout-Aware Markdown
        print(f"[*] Parsing document: {file_path}")
        try:
            raw_markdown = self.converter.convert(file_path).document.export_to_markdown()
        except Exception as e:
            raise RuntimeError(f"Docling failed to parse the document: {e}")

        # STEP 2: Extract Technical Entities using GLiNER
        print("[*] Extracting technical entities to build Allowlist...")
        # We pass the entire markdown text. GLiNER handles it well.
        entities = self.gliner_model.predict_entities(raw_markdown, self.tech_labels)
        
        # Build a structured dictionary for our Graph Database
        structured_data = {label: set() for label in self.tech_labels}
        dynamic_allow_list = set()

        for entity in entities:
            text_val = entity["text"].strip()
            structured_data[entity["label"]].add(text_val)
            dynamic_allow_list.add(text_val) # Add to Presidio's blindspot

        # Convert sets back to lists for JSON serialization
        structured_json = {k: list(v) for k, v in structured_data.items() if v}
        allow_list = list(dynamic_allow_list)
        print(f" └── Found {len(allow_list)} technical terms to protect from redaction.")

        # STEP 3: Redact PII, utilizing the dynamic Allowlist
        print("[*] Scrubbing PII (Names, Contact, Locations)...")
        analyzer_results = self.analyzer.analyze(
            text=raw_markdown, 
            entities=self.pii_entities, 
            language='en',
            allow_list=allow_list 
        )
        
        anonymized_result = self.anonymizer.anonymize(
            text=raw_markdown, 
            analyzer_results=analyzer_results
        )

        return anonymized_result.text, structured_json


# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    pipeline = SecureResumePipeline()
    
    # Targeting the resume file you uploaded
    target_resume = r"C:\Users\ssanj\OneDrive\Documents\HUB_2.0\RES-AI\sample_resume\Resume.pdf"
    
    try:
        clean_markdown, extracted_entities = pipeline.run(target_resume)
        
        print("\n" + "="*50)
        print("✅ PIPELINE EXECUTION SUCCESSFUL")
        print("="*50)
        
        print("\n[1] EXTRACTED STRUCTURED DATA (Ready for Graph DB):")
        print(json.dumps(extracted_entities, indent=4))
        
        print("\n" + "-"*50)
        
        print("\n[2] SANITIZED RESUME TEXT (Ready for LLM Processing):")
        # Printing just the first 1000 characters to verify it worked without flooding the terminal
        print(clean_markdown[:1000] + "\n\n... [Text Truncated for Display] ...")
        
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")