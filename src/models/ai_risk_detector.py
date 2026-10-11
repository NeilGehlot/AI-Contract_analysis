"""Document-level risk detector for the embedding classifier."""

from pathlib import Path
from typing import Dict, List

import yaml

from src.models.fast_ai_risk_classifier import FastAIRiskClassifier
from src.utils.clause_splitter import split_into_clauses
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RECOMMENDATIONS = {
    "HIGH": "Review immediately — negotiate liability caps, scope limits, or termination rights.",
    "MEDIUM": "Review this clause and consider negotiating more balanced terms.",
}


class AIRiskDetector:
    """Runs the trained FastAIRiskClassifier over contract clauses."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        risk_cfg = config.get("risk_detection", {})
        self.min_confidence = risk_cfg.get("min_confidence", 0.5)
        model_path = risk_cfg.get("model_path", "data/models/fast_risk_classifier.joblib")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"AI risk model not found at {model_path}. "
                "Run: python scripts/train_risk_classifier.py"
            )

        self.classifier = FastAIRiskClassifier()
        self.classifier.load(model_path)
        logger.info("AIRiskDetector initialized (ML sentence-embedding classifier)")

    def detect_risks(self, text: str, entities: Dict) -> List[Dict]:
        risks = []
        for clause in split_into_clauses(text):
            prediction = self.classifier.predict(clause)
            if not prediction["flagged"]:
                continue
            if prediction["confidence"] < self.min_confidence:
                continue

            severity = prediction["severity"]
            risks.append(
                {
                    "type": f"ai_{severity.lower()}_risk",
                    "severity": severity,
                    "description": (
                        f"AI classifier flagged {severity} risk "
                        f"(confidence: {prediction['confidence']:.0%})"
                    ),
                    "recommendation": RECOMMENDATIONS.get(
                        severity, "Review with legal counsel."
                    ),
                    "clause": clause[:200] + ("..." if len(clause) > 200 else ""),
                    "confidence": prediction["confidence"],
                    "method": prediction["method"],
                    "probabilities": prediction["probabilities"],
                }
            )

        logger.info(f"AI risk detector flagged {len(risks)} clauses")
        return risks
