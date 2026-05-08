from docling.document_converter import DocumentConverter
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from gliner import GLiNER
import json

class SecureResumePipeline:
    def __init__(self):
        print("1. Loading Docling (PDF -> Markdown)...")
        self.converter = DocumentConverter()
        
        print("2. Loading GLiNER (Zero-Shot Entity Extraction)...")
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        self.tech_labels = [
            "Skill", "Programming Language", "Framework", 
            "Database", "Job Title", "Degree", "University", "Tool"
        ]
        
        print("3. Loading Presidio (PII Redaction Shield)...")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.pii_entities = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]

    def run(self, file_path: str):
        # STEP 1: Extract structure
        print(f"\nParsing {file_path}...")
        raw_markdown = self.converter.convert(file_path).document.export_to_markdown()

        # STEP 2: Extract Technical Terms via GLiNER
        print("Extracting technical entities...")
        entities = self.gliner_model.predict_entities(raw_markdown, self.tech_labels)
        
        # Build our dynamic Allowlist (extract just the text values)
        # e.g., ["Docker", "Linux", "Merkle", "Cybervault"]
        dynamic_allow_list = list(set([entity["text"] for entity in entities]))
        print(f"Allowlist generated with {len(dynamic_allow_list)} technical terms.")

        # STEP 3: Run Presidio WITH the Allowlist
        print("Scrubbing PII...")
        analyzer_results = self.analyzer.analyze(
            text=raw_markdown, 
            entities=self.pii_entities, 
            language='en',
            allow_list=dynamic_allow_list  # <--- The magic happens here
        )
        
        safe_markdown = self.anonymizer.anonymize(
            text=raw_markdown, 
            analyzer_results=analyzer_results
        ).text

        # Return both the clean text (for the LLM) and the structured JSON (for the Graph DB)
        structured_skills = {label: set() for label in self.tech_labels}
        for entity in entities:
            structured_skills[entity["label"]].add(entity["text"])
        structured_skills = {k: list(v) for k, v in structured_skills.items()}

        return safe_markdown, structured_skills

# --- Run the Pipeline ---
if __name__ == "__main__":
    pipeline = SecureResumePipeline()
    # Replace with your actual PDF
    clean_text, json_data = pipeline.run("C:/Users/ssanj/OneDrive/Documents/HUB_2.0/RES-AI/sample_resume/Resume.pdf") 
    
    print("\n--- EXTRACTED JSON FOR NEO4J ---")
    print(json.dumps(json_data, indent=2))