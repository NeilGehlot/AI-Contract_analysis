"""
Risk Detector
Detects potential risks in contracts
"""

import re
from typing import List, Dict
import yaml
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class RiskDetector:
    """Detect risks in contracts"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize risk detector"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.rules = config['risk_rules']
        logger.info("RiskDetector initialized")
    
    def detect_risks(self, text: str, entities: Dict) -> List[Dict]:
        """
        Detect risks in contract
        
        Args:
            text: Contract text
            entities: Extracted entities
            
        Returns:
            List of detected risks
        """
        logger.info("Detecting risks...")
        
        risks = []
        text_lower = text.lower()
        
        for keyword in self.rules['liability_keywords']:
            if keyword in text_lower:
                risks.append({
                    'type': 'unlimited_liability',
                    'severity': 'HIGH',
                    'description': f'Contract contains "{keyword}"',
                    'recommendation': 'Negotiate a liability cap (typically 1-2x contract value)',
                    'clause': self._extract_clause(text, keyword)
                })
        
        payment_terms = entities.get('terms', {}).get('payment_terms', '')
        if payment_terms:
            days = int(''.join(filter(str.isdigit, payment_terms)))
            if days > self.rules['payment_terms_threshold']:
                risks.append({
                    'type': 'long_payment_terms',
                    'severity': 'MEDIUM',
                    'description': f'{days}-day payment terms (standard: 30-45 days)',
                    'recommendation': 'Negotiate to reduce payment terms to 45 days',
                    'clause': f'Payment terms: {days} days'
                })
        
        for clause in self.rules['missing_clauses']:
            if clause.lower() not in text_lower:
                risks.append({
                    'type': 'missing_clause',
                    'severity': 'MEDIUM',
                    'description': f'Missing {clause} clause',
                    'recommendation': f'Request addition of {clause} clause',
                    'clause': None
                })
        
        for term in self.rules['unfair_terms']:
            if term in text_lower:
                risks.append({
                    'type': 'unfair_term',
                    'severity': 'HIGH',
                    'description': f'Contains potentially unfair term: {term}',
                    'recommendation': 'Review and negotiate this clause',
                    'clause': self._extract_clause(text, term)
                })
        
        logger.info(f"Detected {len(risks)} risks")
        return risks
    
    def _extract_clause(self, text: str, keyword: str) -> str:
        """Extract clause containing keyword"""
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                return sentence.strip()[:200] + "..."
        return "Clause not extracted"