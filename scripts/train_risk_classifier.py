"""Train the sentence-embedding plus logistic regression risk classifier.

    Run from the repo root after the CUAD training JSON exists:
        python scripts/train_risk_classifier.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fast_ai_risk_classifier import FastAIRiskClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the sentence-embedding plus logistic regression risk classifier"
    )
    parser.parse_args()
    training_path = Path("data/processed/risk_training_data.json")
    if not training_path.exists():
        raise FileNotFoundError(
            f"{training_path} not found. Run scripts/build_training_data_from_cuad.py first."
        )

    with open(training_path, encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    model = FastAIRiskClassifier()
    model.train(texts, labels)
    model.save("data/models/fast_risk_classifier.joblib")
    print("Saved to data/models/fast_risk_classifier.joblib")


if __name__ == "__main__":
    main()
