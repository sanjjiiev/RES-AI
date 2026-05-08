import os
from docling.document_converter import DocumentConverter
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class ResumeIngestionPipeline:
    def __init__(self):
        print("Initializing Docling Converter...")
        self.converter = DocumentConverter()
        
        print("Initializing Microsoft Presidio Shield...")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # We specifically target demographics and contact info to prevent bias/leaks
        self.pii_entities = [
            "PERSON", 
            "PHONE_NUMBER", 
            "EMAIL_ADDRESS", 
            "LOCATION"
        ]

    def parse_document(self, file_path: str) -> str:
        """Extracts text and layout from PDF using Docling."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found at {file_path}")
            
        print(f"Parsing document: {file_path}")
        result = self.converter.convert(file_path)
        
        # Exporting to Markdown preserves headers, bullet points, and tables perfectly
        return result.document.export_to_markdown()

    def sanitize_text(self, raw_text: str) -> str:
        """Scrub PII from the text to ensure secure, unbiased AI evaluation."""
        print("Scanning for PII...")
        
        # Detect the entities
        analyzer_results = self.analyzer.analyze(
            text=raw_text, 
            entities=self.pii_entities, 
            language='en'
        )
        
        # Mask the detected entities (e.g., replaces "John Doe" with "<PERSON>")
        anonymized_result = self.anonymizer.anonymize(
            text=raw_text, 
            analyzer_results=analyzer_results
        )
        
        return anonymized_result.text

    def run_pipeline(self, file_path: str) -> str:
        """The main execution flow."""
        raw_markdown = self.parse_document(file_path)
        clean_markdown = self.sanitize_text(raw_markdown)
        return clean_markdown

# --- Testing the Pipeline ---
if __name__ == "__main__":
    # Create a dummy PDF path to test (Make sure to place a real resume PDF here)
    sample_resume = r"C:\Users\ssanj\OneDrive\Documents\HUB_2.0\RES-AI\sample_resume\Resume.pdf"
    
    pipeline = ResumeIngestionPipeline()
    
    try:
        safe_resume_text = pipeline.run_pipeline(sample_resume)
        print("\n--- SECURE, EXTRACTED RESUME DATA ---")
        print(safe_resume_text)
    except Exception as e:
        print(f"Error: {e}")