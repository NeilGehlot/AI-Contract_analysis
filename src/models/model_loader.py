"""Factory helpers for loading model components."""

from pathlib import Path

import yaml

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_entity_extractor(config_path: str = "config.yaml"):
    config = load_config(config_path)
    ner_cfg = config.get("entity_extraction", {})
    engine = ner_cfg.get("engine", "spacy")
    model_path = ner_cfg.get("model_path", "data/models/ner")

    if engine == "legal_bert":
        from src.models.legal_bert_ner import LegalBertEntityExtractor

        if LegalBertEntityExtractor.is_available(model_path):
            return LegalBertEntityExtractor(model_path)
        logger.warning(
            f"LEGAL-BERT NER not found at {model_path}; falling back to spaCy. "
            "Run: python scripts/train_legal_bert_ner.py"
        )

    from src.models.entity_extractor import EntityExtractor
    return EntityExtractor()


def load_risk_detector(config_path: str = "config.yaml"):
    config = load_config(config_path)
    risk_cfg = config.get("risk_detection", {})
    engine = risk_cfg.get("engine", "fast_ai")

    if engine == "legal_bert":
        from src.models.legal_bert_risk_detector import LegalBertRiskDetector
        from src.models.legal_bert_risk_classifier import LegalBertRiskClassifier

        model_path = risk_cfg.get(
            "legal_bert_model_path", "data/models/risk_classifier_final"
        )
        if LegalBertRiskClassifier.is_available(model_path):
            return LegalBertRiskDetector(config_path)
        logger.warning(
            f"LEGAL-BERT risk model not found at {model_path}; falling back to fast_ai."
        )
        engine = "fast_ai"

    if engine == "fast_ai":
        from src.models.ai_risk_detector import AIRiskDetector
        return AIRiskDetector(config_path)

    from src.models.risk_detector import RiskDetector
    return RiskDetector(config_path)
