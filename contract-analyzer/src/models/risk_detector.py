import re
import yaml
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
HIGH = "HIGH"
MEDIUM = "MEDIUM"

UNLIMITED_LIABILITY = "unlimited_liability"
LONG_PAYMENT_TERMS = "long_payment_terms"
MISSING_CLAUSE = "missing_clause"
UNFAIR_TERM = "unfair_term"



class RiskDetector:
    """
    Detect risks in contracts
    """
    def _create_risk(
        self,
        risk_type,
        severity,
        description,
        recommendation,
        clause,
        ):  
        return {
            "type": risk_type,
            "severity": severity,
            "description": description,
            "recommendation": recommendation,
            "clause": clause,
        }
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize risk detector
        """
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            self.rules = config['risk_rules']
            logger.info("RiskDetector initialized")
    
    def detect_risks(self, text, entities) :
        """
        Detect risks in contract
        """
        logger.info("Detecting risks...")
        
        risks = []
        text_lower = text.lower()
        
        for keyword in self.rules['liability_keywords']:
            if keyword in text_lower:
                risks.append(
                    self._create_risk(
                        UNLIMITED_LIABILITY,
                        HIGH,
                        f'Contract contains "{keyword}"',
                        "Negotiate a liability cap (typically 1-2x contract value)",
                        self._extract_clause(text, keyword)
                    )
                )
        
        payment_terms = entities.get('terms', {}).get('payment_terms', '')
        if payment_terms:
            days = int(''.join(filter(str.isdigit, payment_terms)))
            if days > self.rules['payment_terms_threshold']:
                risks.append(
                    self._create_risk(
                        LONG_PAYMENT_TERMS,
                        MEDIUM,
                        f"{days}-day payment terms (standard: 30-45 days)",
                        "Negotiate to reduce payment terms to 45 days",
                        f"Payment terms: {days} days"
                    )
                )
        
        for clause in self.rules['missing_clauses']:
            if clause.lower() not in text_lower:
                risks.append(
                    self._create_risk(
                        MISSING_CLAUSE,
                        MEDIUM,
                        f"Missing {clause} clause",
                        f"Request addition of {clause} clause",
                        None
                    )
                )
        
        for term in self.rules['unfair_terms']:
            if term in text_lower:
                risks.append(
                    self._create_risk(
                        UNFAIR_TERM,
                        HIGH,
                        f"Contains potentially unfair term: {term}",
                        "Review and negotiate this clause",
                        self._extract_clause(text, term)
                    )
                )        
        logger.info(f"Detected {len(risks)} risks")
        return risks
    
    def _extract_clause(self, text: str, keyword: str):
        """
        Extract clause containing keyword
        """
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                sentence = sentence.strip()

                return (
                    sentence[:200] + "..."
                    if len(sentence) > 200
                    else sentence
                )
        return "Clause not extracted"