"""
Entity Extractor
Extracts key entities from contracts using spaCy
"""

import spacy
import re
from typing import List, Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class EntityExtractor:
    """Extract entities from contract text"""
    
    def __init__(self):
        """Initialize entity extractor"""
        logger.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise
    
    def extract_all(self, text: str) -> Dict:
        """
        Extract all entities
        
        Args:
            text: Contract text
            
        Returns:
            Dictionary of extracted entities
        """
        logger.info("Extracting entities...")
        
        return {
            'parties': self.extract_parties(text),
            'dates': self.extract_dates(text),
            'amounts': self.extract_amounts(text),
            'emails': self.extract_emails(text),
            'terms': self.extract_contract_terms(text)
        }
    
    def extract_parties(self, text: str) -> List[str]:
        """Extract party names"""
        parties = []
        
        # Look for BETWEEN clause
        between_match = re.search(
            r'(?:BETWEEN|between)\s+(.*?)\s+(?:AND|and)\s+(.*?)(?:\.|,|\()',
            text,
            re.DOTALL
        )
        
        if between_match:
            party1 = re.sub(r'\(.*?\)', '', between_match.group(1)).strip()
            party2 = re.sub(r'\(.*?\)', '', between_match.group(2)).strip()
            parties.extend([party1, party2])
        
        # Use spaCy NER
        doc = self.nlp(text[:5000])
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON'] and ent.text not in parties:
                if len(ent.text) > 3:
                    parties.append(ent.text)
        
        # Remove duplicates
        parties = list(dict.fromkeys(parties))[:10]
        logger.info(f"Found {len(parties)} parties")
        return parties
    
    def extract_dates(self, text: str) -> List[Dict]:
        """Extract dates"""
        dates = []
        
        patterns = [
            (r'dated\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})', 'signing_date'),
            (r'effective\s+(?:from|as of)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})', 'effective_date'),
            (r'expir(?:y|ing|es)\s+(?:on|date)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})', 'expiry_date'),
            (r'(\d{1,2}\s+\w+\s+\d{4})', 'general_date')
        ]
        
        for pattern, date_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append({
                    'date': match.group(1),
                    'type': date_type
                })
        
        logger.info(f"Found {len(dates)} dates")
        return dates[:10]
    
    def extract_amounts(self, text: str) -> List[Dict]:
        """Extract monetary amounts"""
        amounts = []
        
        patterns = [
            (r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', '₹'),
            (r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)', '₹'),
            (r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', '$')
        ]
        
        for pattern, currency in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount_value = float(amount_str)
                    amounts.append({
                        'amount': match.group(1),
                        'currency': currency,
                        'value': amount_value
                    })
                except ValueError:
                    pass
        
        amounts.sort(key=lambda x: x['value'], reverse=True)
        logger.info(f"Found {len(amounts)} amounts")
        return amounts[:10]
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = list(set(re.findall(pattern, text)))[:5]
        logger.info(f"Found {len(emails)} emails")
        return emails
    
    def extract_contract_terms(self, text: str) -> Dict:
        """Extract contract terms"""
        terms = {}
        
        # Duration
        term_match = re.search(
            r'term\s+of\s+(\d+)\s+(year|month|day)s?',
            text,
            re.IGNORECASE
        )
        if term_match:
            terms['duration'] = f"{term_match.group(1)} {term_match.group(2)}(s)"
        
        # Notice period
        notice_match = re.search(
            r'notice\s+(?:period\s+)?of\s+(\d+)\s+(day|month)s?',
            text,
            re.IGNORECASE
        )
        if notice_match:
            terms['notice_period'] = f"{notice_match.group(1)} {notice_match.group(2)}(s)"
        
        # Payment terms
        payment_match = re.search(
            r'payment.*?(\d+)\s+days?',
            text,
            re.IGNORECASE
        )
        if payment_match:
            terms['payment_terms'] = f"{payment_match.group(1)} days"
        
        logger.info(f"Extracted {len(terms)} contract terms")
        return terms