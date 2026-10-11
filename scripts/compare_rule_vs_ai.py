"""Compare the rule engine with the embedding classifier on a clause-level holdout.

    Run from the repo root:
        python scripts/compare_rule_vs_ai.py
"""

import torch  # preload on Windows before sentence_transformers

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.models.fast_ai_risk_classifier import FastAIRiskClassifier
from src.rules.risk_engine import RuleBasedRiskEngine


def load_holdout_test(
    data_path: str,
    test_size: float = 0.15,
    source_filter: str = "cuad_human",
) -> tuple[list[str], list[str]]:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    if source_filter:
        data = [row for row in data if row.get("source") == source_filter]

    texts = [row["text"] for row in data]
    labels = [row["label"] for row in data]
    if len(texts) < 20:
        raise ValueError(f"Need at least 20 examples, found {len(texts)}")

    _, test_texts, _, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=42,
    )
    return test_texts, test_labels


def summarize(y_true: list[str], y_pred: list[str]) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    return {
        "weighted_precision": round(float(p), 3),
        "weighted_recall": round(float(r), 3),
        "weighted_f1": round(float(f1), 3),
        "per_class": per_class,
    }


def rule_predict(rule_engine: RuleBasedRiskEngine, text: str) -> str:
    result = rule_engine.check_clause(text)
    if result["flagged"]:
        return result["severity"]
    return "LOW"


def compare(
    data_path: str,
    output_dir: str = "outputs/reports",
    source_filter: str = "cuad_human",
) -> dict:
    test_texts, test_labels = load_holdout_test(data_path, source_filter=source_filter)

    rule_engine = RuleBasedRiskEngine()
    ai_model = FastAIRiskClassifier()
    ai_model.load("data/models/fast_risk_classifier.joblib")

    rule_preds = [rule_predict(rule_engine, t) for t in test_texts]
    ai_preds = [ai_model.predict(t)["severity"] for t in test_texts]

    rule_metrics = summarize(test_labels, rule_preds)
    ai_metrics = summarize(test_labels, ai_preds)

    results = {
        "test_set_size": len(test_texts),
        "label_source": source_filter or "all",
        "rule_based": rule_metrics,
        "ai_classifier": ai_metrics,
        "comparison_table": [
            {
                "method": "Rule-based (Sem 3/4)",
                "precision": rule_metrics["weighted_precision"],
                "recall": rule_metrics["weighted_recall"],
                "f1": rule_metrics["weighted_f1"],
            },
            {
                "method": "AI classifier (Sem 5)",
                "precision": ai_metrics["weighted_precision"],
                "recall": ai_metrics["weighted_recall"],
                "f1": ai_metrics["weighted_f1"],
            },
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "rule_vs_ai_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = out_dir / "rule_vs_ai_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Rule Engine vs AI Classifier Comparison\n\n")
        f.write(f"- Test clauses: **{len(test_texts)}** (held-out 15%, source: `{source_filter}`)\n\n")
        f.write("| Method | Precision | Recall | F1 |\n")
        f.write("|--------|-----------|--------|----|\n")
        for row in results["comparison_table"]:
            f.write(
                f"| {row['method']} | {row['precision']:.1%} | "
                f"{row['recall']:.1%} | {row['f1']:.3f} |\n"
            )

    print(f"Test set: {len(test_texts)} clauses (source={source_filter})")
    print("\n=== RULE-BASED (Sem 3/4) ===")
    print(classification_report(test_labels, rule_preds, zero_division=0))
    print(
        f"Weighted: precision={rule_metrics['weighted_precision']:.3f} "
        f"recall={rule_metrics['weighted_recall']:.3f} "
        f"f1={rule_metrics['weighted_f1']:.3f}\n"
    )

    print("=== AI CLASSIFIER (Sem 5) ===")
    print(classification_report(test_labels, ai_preds, zero_division=0))
    print(
        f"Weighted: precision={ai_metrics['weighted_precision']:.3f} "
        f"recall={ai_metrics['weighted_recall']:.3f} "
        f"f1={ai_metrics['weighted_f1']:.3f}\n"
    )

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rule engine vs AI risk classifier")
    parser.add_argument(
        "--data",
        default="data/processed/risk_training_data.json",
        help="Training data JSON (holdout split applied internally)",
    )
    parser.add_argument(
        "--source",
        default="cuad_human",
        help="Only evaluate examples with this source tag (use cuad_human for expert labels)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
    )
    args = parser.parse_args()
    compare(args.data, args.output_dir, args.source)


if __name__ == "__main__":
    main()
