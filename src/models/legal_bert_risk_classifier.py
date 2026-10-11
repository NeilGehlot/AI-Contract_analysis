"""LEGAL-BERT sequence classifier for clause-level risk severity (Sem 6)."""

import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ID2LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


class LegalBertRiskClassifier:
    """Fine-tuned LEGAL-BERT risk classifier."""

    def __init__(self, model_path: str = "data/models/risk_classifier_final"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, clause_text: str) -> dict:
        inputs = self.tokenizer(
            clause_text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        severity = ID2LABEL[pred_id]
        return {
            "flagged": severity != "LOW",
            "severity": severity,
            "confidence": round(float(probs[pred_id]), 3),
            "method": "legal_bert_classifier",
            "probabilities": {
                ID2LABEL[i]: round(float(p), 3) for i, p in enumerate(probs)
            },
        }

    @classmethod
    def is_available(cls, model_path: str = "data/models/risk_classifier_final") -> bool:
        path = Path(model_path)
        return path.exists() and (path / "config.json").exists()
