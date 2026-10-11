"""Document-level LEGAL-BERT risk detector."""

from pathlib import Path
from typing import Dict, List

import yaml

from src.models.legal_bert_risk_classifier import LegalBertRiskClassifier
from src.utils.clause_splitter import split_into_clauses
from src.utils.risk_explanation import explain_clause
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RECOMMENDATIONS = {
    "HIGH": "Review immediately — negotiate liability caps, scope limits, or termination rights.",
    "MEDIUM": "Review this clause and consider negotiating more balanced terms.",
}


class LegalBertRiskDetector:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        risk_cfg = config.get("risk_detection", {})
        self.min_confidence = risk_cfg.get("min_confidence", 0.5)
        model_path = risk_cfg.get(
            "legal_bert_model_path", "data/models/risk_classifier_final"
        )

        if not LegalBertRiskClassifier.is_available(model_path):
            raise FileNotFoundError(
                f"LEGAL-BERT risk model not found at {model_path}. "
                "Run: python scripts/train_legal_bert_risk.py"
            )

        self.classifier = LegalBertRiskClassifier(model_path)
        logger.info("LegalBertRiskDetector initialized")

    def detect_risks(self, text: str, entities: Dict) -> List[Dict]:
        risks = []
        for clause in split_into_clauses(text):
            prediction = self.classifier.predict(clause)
            if not prediction["flagged"] or prediction["confidence"] < self.min_confidence:
                continue

            severity = prediction["severity"]
            note = explain_clause(clause, severity)
            shown = clause if len(clause) <= 500 else clause[:500] + "..."
            risks.append(
                {
                    "type": f"bert_{severity.lower()}_risk",
                    "severity": severity,
                    "topic": note["topic"],
                    "description": note["consideration"],
                    "consideration": note["consideration"],
                    "recommendation": note["recommendation"],
                    "clause": shown,
                    "confidence": prediction["confidence"],
                    "method": prediction["method"],
                    "probabilities": prediction["probabilities"],
                }
            )

        logger.info(f"LEGAL-BERT risk detector flagged {len(risks)} clauses")
        return risks
