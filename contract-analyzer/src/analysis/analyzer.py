"""
Contract Analyzer
Main analysis orchestrator
"""

from pathlib import Path
from datetime import datetime
from typing import Dict

from src.utils.pdf_extractor import PDFExtractor
from src.utils.text_cleaner import TextCleaner
from src.models.entity_extractor import EntityExtractor
from src.models.contract_classifier import ContractClassifier
from src.models.risk_detector import RiskDetector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ContractAnalyzer:
    """Main contract analyzer"""
    
    def __init__(self):
        """Initialize analyzer"""
        logger.info("="*60)
        logger.info("Initializing Contract Analyzer")
        logger.info("="*60)
        
        self.pdf_extractor = PDFExtractor()
        self.text_cleaner = TextCleaner()
        self.entity_extractor = EntityExtractor()
        self.classifier = ContractClassifier()
        self.risk_detector = RiskDetector()
        
        logger.info("✓ All components initialized successfully")
    
    def analyze(self, pdf_path: str) -> Dict:
        """
        Analyze contract
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Analysis results
        """
        logger.info("\n" + "="*60)
        logger.info(f"ANALYZING: {Path(pdf_path).name}")
        logger.info("="*60 + "\n")
        
        # Step 1: Extract text
        logger.info("Step 1/5: Extracting text from PDF...")
        pdf_result = self.pdf_extractor.extract_text(pdf_path)
        
        if not pdf_result['success']:
            logger.error("Failed to extract text")
            return {'success': False, 'error': 'Text extraction failed'}
        
        raw_text = pdf_result['text']
        logger.info(f"  ✓ Extracted {len(raw_text)} characters from {pdf_result['pages']} pages")
        
        # Step 2: Clean text
        logger.info("\nStep 2/5: Cleaning text...")
        clean_text = self.text_cleaner.clean(raw_text)
        stats = self.text_cleaner.get_statistics(clean_text)
        logger.info(f"  ✓ Cleaned: {stats['word_count']} words, {stats['sentence_count']} sentences")
        
        # Step 3: Extract entities
        logger.info("\nStep 3/5: Extracting entities...")
        entities = self.entity_extractor.extract_all(clean_text)
        logger.info(f"  ✓ Found: {len(entities['parties'])} parties, "
                   f"{len(entities['dates'])} dates, {len(entities['amounts'])} amounts")
        
        # Step 4: Classify contract
        logger.info("\nStep 4/5: Classifying contract type...")
        classification = self.classifier.classify(clean_text)
        logger.info(f"  ✓ Type: {classification['type']} "
                   f"(confidence: {classification['confidence']:.2f})")
        
        # Step 5: Detect risks
        logger.info("\nStep 5/5: Detecting risks...")
        risks = self.risk_detector.detect_risks(clean_text, entities)
        logger.info(f"  ✓ Identified {len(risks)} potential risks")
        
        # Compile results
        result = {
            'success': True,
            'file_info': {
                'filename': Path(pdf_path).name,
                'pages': pdf_result['pages'],
                'analyzed_at': datetime.now().isoformat()
            },
            'text_stats': stats,
            'classification': classification,
            'entities': entities,
            'risks': risks,
            'sample_text': clean_text[:500] + "..."
        }
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60 + "\n")
        
        return result