"""Bootstrap risk training rows from local PDFs using the rule engine.

    Run from the repo root:
        python scripts/build_risk_training_data.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rules.risk_engine import RuleBasedRiskEngine
from src.utils.pdf_extractor import PDFExtractor
from src.utils.text_cleaner import TextCleaner


def split_into_clauses(text: str) -> list[dict]:
    """Split contract text into clause-sized chunks for labeling."""
    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for paragraph in paragraphs:
        if len(paragraph) > 400:
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", paragraph)
                if len(s.strip()) > 40
            ]
            chunks.extend(sentences or [paragraph])
        elif len(paragraph) > 30:
            chunks.append(paragraph)

    if not chunks:
        chunks = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 40
        ]

    return [{"text": c, "type": "UNKNOWN"} for c in chunks]


def build_training_examples(annotated_contracts_path: str, output_path: str) -> dict:
    with open(annotated_contracts_path, encoding="utf-8") as f:
        contracts = json.load(f)

    rule_engine = RuleBasedRiskEngine()
    examples = []

    for contract in contracts:
        for clause in contract["clauses"]:
            text = clause["text"]
            human_label = clause.get("risk_label")
            rule_result = rule_engine.check_clause(text, clause.get("type"))
            if human_label:
                final_label = human_label
                source = "human"
            else:
                final_label = rule_result["severity"] if rule_result["flagged"] else "LOW"
                source = "rule"

            examples.append(
                {
                    "text": text,
                    "clause_type": clause.get("type", "UNKNOWN"),
                    "label": final_label,
                    "source": source,
                }
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    stats = {
        "total": len(examples),
        "human": sum(1 for e in examples if e["source"] == "human"),
        "rule": sum(1 for e in examples if e["source"] == "rule"),
    }
    print(
        f"Built {stats['total']} examples "
        f"({stats['human']} human-labeled, {stats['rule']} rule-labeled)"
    )
    return stats


def bootstrap_annotated_from_pdfs(raw_dir: str, output_path: str) -> str:
    """Build annotated_contracts.json from PDFs when the full annotation set is unavailable."""
    extractor = PDFExtractor()
    cleaner = TextCleaner()
    contracts = []

    for pdf_path in sorted(Path(raw_dir).glob("*.pdf")):
        result = extractor.extract_text(str(pdf_path))
        if not result["success"]:
            print(f"Skipping {pdf_path.name}: extraction failed")
            continue
        text = cleaner.clean(result["text"])
        contracts.append(
            {
                "filename": pdf_path.name,
                "clauses": split_into_clauses(text),
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2)

    clause_count = sum(len(c["clauses"]) for c in contracts)
    print(f"Bootstrapped {len(contracts)} contracts / {clause_count} clauses -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build risk training data for Sem 5 AI classifier")
    parser.add_argument(
        "--input",
        default="data/processed/annotated_contracts.json",
        help="Annotated contracts JSON (contracts with clauses[] and optional risk_label)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/risk_training_data.json",
        help="Output training examples JSON",
    )
    parser.add_argument(
        "--bootstrap-from-pdfs",
        action="store_true",
        help="Create annotated_contracts.json from PDFs in data/raw first",
    )
    parser.add_argument(
        "--force-bootstrap",
        action="store_true",
        help="Rebuild annotated_contracts.json from PDFs even if it already exists",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="PDF directory used with --bootstrap-from-pdfs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.bootstrap_from_pdfs and (args.force_bootstrap or not input_path.exists()):
        bootstrap_annotated_from_pdfs(args.raw_dir, str(input_path))
    elif not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Copy your 1,512 annotated contracts there, "
            "or rerun with --bootstrap-from-pdfs to use local sample PDFs."
        )

    build_training_examples(str(input_path), args.output)


if __name__ == "__main__":
    main()
