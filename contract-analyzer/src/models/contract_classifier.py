"""
Contract Classifier
Classifies contract type using keyword matching
"""

import re
from typing import Dict
import yaml
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ContractClassifier:
    """Classify contract type"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize classifier"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.contract_types = config['contract_types']
        
        # Keywords for each type
        self.keywords = {
            'NDA': ['non-disclosure', 'confidential', 'nda', 'secrecy'],
            'Employment Agreement': ['employment', 'employee', 'employer', 'job', 'position'],
            'Service Agreement': ['service', 'services', 'scope of work', 'deliverables'],
            'Vendor Agreement': ['vendor', 'supplier', 'purchase', 'procurement'],
            'Consulting Agreement': ['consulting', 'consultant', 'advisory', 'professional services'],
            'Lease Agreement': ['lease', 'rent', 'tenant', 'landlord', 'premises'],
            'Partnership Agreement': ['partnership', 'partner', 'joint venture'],
            'Sales Agreement': ['sale', 'purchase', 'buyer', 'seller']
        }
        
        logger.info("ContractClassifier initialized")
    
    def classify(self, text: str) -> Dict:
        """
        Classify contract type
        
        Args:
            text: Contract text
            
        Returns:
            Dictionary with classification results
        """
        logger.info("Classifying contract type...")
        
        text_lower = text.lower()
        scores = {}
        
        for contract_type, keywords in self.keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[contract_type] = score
        
        # Get top type
        if scores:
            top_type = max(scores, key=scores.get)
            confidence = scores[top_type] / len(self.keywords[top_type])
            
            logger.info(f"Classified as: {top_type} (confidence: {confidence:.2f})")
            
            return {
                'type': top_type,
                'confidence': confidence,
                'all_scores': scores
            }
        
        return {
            'type': 'Unknown',
            'confidence': 0.0,
            'all_scores': {}
        }