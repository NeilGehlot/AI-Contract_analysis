"""Sem 5 AI risk classifier — sentence embeddings + Logistic Regression."""

import torch  # preload on Windows before sentence_transformers

import json
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class FastAIRiskClassifier:
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embed_model)
        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial",
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def train(
        self,
        texts: list,
        labels: list,
        test_size: float = 0.15,
        show_progress_bar: bool = True,
    ) -> str:
        y = self.label_encoder.fit_transform(labels)
        print(f"Embedding {len(texts)} clauses...")
        X = self.embedder.encode(texts, show_progress_bar=show_progress_bar, batch_size=64)
        if test_size <= 0:
            self.clf.fit(X, y)
            self.is_fitted = True
            print(f"Fitted on all {len(texts)} examples (test_size={test_size}).")
            return ""
        split_kwargs = {"test_size": test_size, "random_state": 42}
        if len(set(labels)) > 1 and len(labels) >= 10:
            split_kwargs["stratify"] = y
        X_train, X_val, y_train, y_val = train_test_split(X, y, **split_kwargs)
        self.clf.fit(X_train, y_train)
        self.is_fitted = True
        preds = self.clf.predict(X_val)
        report = classification_report(
            y_val,
            preds,
            labels=range(len(self.label_encoder.classes_)),
            target_names=list(self.label_encoder.classes_),
            zero_division=0,
        )
        print(report)
        return report

    def predict(self, clause_text: str) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Model not trained or loaded.")
        embedding = self.embedder.encode([clause_text])
        pred_id = self.clf.predict(embedding)[0]
        probs = self.clf.predict_proba(embedding)[0]
        severity = self.label_encoder.inverse_transform([pred_id])[0]
        return {
            "flagged": severity != "LOW",
            "severity": severity,
            "confidence": round(float(max(probs)), 3),
            "method": "ai_embedding_classifier",
            "probabilities": {
                cls: round(float(p), 3)
                for cls, p in zip(self.label_encoder.classes_, probs)
            },
        }

    def check_clause(self, clause_text: str, clause_type: str = None) -> dict:
        """Drop-in alias matching RuleBasedRiskEngine interface."""
        result = self.predict(clause_text)
        return {
            "flagged": result["flagged"],
            "reason": f"AI severity: {result['severity']}",
            "severity": result["severity"],
            "confidence": result["confidence"],
            "method": result["method"],
        }

    def save(self, path: str = "data/models/fast_risk_classifier.joblib") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "label_encoder": self.label_encoder}, path)

    def load(self, path: str = "data/models/fast_risk_classifier.joblib") -> None:
        data = joblib.load(path)
        self.clf = data["clf"]
        self.label_encoder = data["label_encoder"]
        self.is_fitted = True


if __name__ == "__main__":
    training_path = Path("data/processed/risk_training_data.json")
    with open(training_path, encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    model = FastAIRiskClassifier()
    model.train(texts, labels)
    model.save()
    print("Saved to data/models/fast_risk_classifier.joblib")
