"""Download CUAD and write one training row per answer span.

Run from the repo root:
    python scripts/build_training_data_from_cuad.py
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CUAD_JSON_URL = (
    "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"
)


def normalize_question(question: str) -> str:
    """CUAD questions look like: Highlight the parts ... related to \"Uncapped Liability\""""
    match = re.search(r'related to "([^"]+)"', question)
    if match:
        return match.group(1)
    match = re.search(r'parties to the "([^"]+)"', question)
    if match:
        return match.group(1)
    return question


def download_cuad(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / "CUAD_v1.json"
    if json_path.exists():
        return json_path

    print(f"Downloading CUAD JSON from HuggingFace ({CUAD_JSON_URL}) ...")
    urlretrieve(CUAD_JSON_URL, json_path)
    return json_path


def category_key(name: str) -> str:
    """Lowercase and collapse whitespace so category titles match despite case and spacing."""
    return " ".join(name.strip().lower().split())


# Real CUAD question titles. Matching is via category_key, not exact case.
# Perpetual License and Cap on Liability (incl. carve-outs) are omitted: they are not in CUAD.
CATEGORY_SEVERITY = {
    "Uncapped Liability": "HIGH",
    "Liquidated Damages": "HIGH",
    "Non-Compete": "HIGH",
    "No-Solicit Of Employees": "HIGH",
    "No-Solicit Of Customers": "HIGH",
    "Exclusivity": "HIGH",
    "Covenant Not To Sue": "HIGH",
    "Renewal Term": "MEDIUM",
    "Notice Period To Terminate Renewal": "MEDIUM",
    "Anti-Assignment": "MEDIUM",
    "Termination For Convenience": "MEDIUM",
    "Change Of Control": "MEDIUM",
    "Competitive Restriction Exception": "MEDIUM",
    "Revenue/Profit Sharing": "MEDIUM",
    "Price Restrictions": "MEDIUM",
    "Minimum Commitment": "MEDIUM",
    "Volume Restriction": "MEDIUM",
    "Post-Termination Services": "MEDIUM",
    "Third Party Beneficiary": "MEDIUM",
    "Audit Rights": "MEDIUM",
    "Insurance": "MEDIUM",
    "Warranty Duration": "MEDIUM",
    "Irrevocable Or Perpetual License": "MEDIUM",
    "Non-Disparagement": "MEDIUM",
    "Most Favored Nation": "MEDIUM",
    "Document Name": "LOW",
    "Parties": "LOW",
    "Agreement Date": "LOW",
    "Effective Date": "LOW",
    "Expiration Date": "LOW",
    "Governing Law": "LOW",
    "License Grant": "LOW",
    "Ip Ownership Assignment": "LOW",
    "Source Code Escrow": "LOW",
    "Cap On Liability": "LOW",
}

EXCLUDED_CATEGORIES = (
    "Rofr/Rofo/Rofn",
    "Non-Transferable License",
    "Affiliate License-Licensee",
    "Affiliate License-Licensor",
    "Joint Ip Ownership",
    "Unlimited/All-You-Can-Eat-License",
)


def category_severity(category: str) -> str | None:
    return {category_key(name): label for name, label in CATEGORY_SEVERITY.items()}.get(
        category_key(category)
    )


def raw_category_span_counts(cuad_json_path: Path) -> Counter:
    with open(cuad_json_path, encoding="utf-8") as f:
        data = json.load(f)
    counts: Counter = Counter()
    for item in data["data"]:
        for paragraph in item.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                category = normalize_question(qa.get("question", ""))
                for answer in qa.get("answers") or []:
                    counts[category] += 1
    return counts


def check_mapped_categories(span_counts: Counter) -> None:
    raw_keys = {category_key(name): name for name in span_counts}
    missing = []
    for name in CATEGORY_SEVERITY:
        if category_key(name) not in raw_keys:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Mapped categories with no raw CUAD match after lowercasing and collapsing whitespace: "
            + ", ".join(missing)
        )
    print("Unmapped raw categories (answer spans):")
    for name, count in sorted(span_counts.items(), key=lambda item: (-item[1], item[0])):
        if category_severity(name) is None:
            print(f"  {count:5}  {name}")


def span_examples_from_cuad(cuad_json_path: Path) -> list[dict]:
    """One row per CUAD answer span, with the contract title and that span's category.

    The label is the severity of this span's category.
    Duplicate spans are dropped only inside the same contract.
    """
    span_counts = raw_category_span_counts(cuad_json_path)
    check_mapped_categories(span_counts)

    with open(cuad_json_path, encoding="utf-8") as f:
        data = json.load(f)

    examples: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for item in data["data"]:
        contract_id = (item.get("title") or "").strip() or "UNKNOWN"
        span_index = 0
        for paragraph in item.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                category = normalize_question(qa.get("question", ""))
                severity = category_severity(category)
                if severity is None:
                    continue
                for answer in qa.get("answers") or []:
                    text = (answer.get("text") or "").strip()
                    if len(text) < 20:
                        continue
                    key = (contract_id, category, text[:200])
                    if key in seen:
                        continue
                    seen.add(key)
                    span_index += 1
                    examples.append(
                        {
                            "clause_id": f"{contract_id}::{span_index}",
                            "contract_id": contract_id,
                            "cuad_category": category,
                            "text": text,
                            "label": severity,
                            "source": "cuad_human",
                        }
                    )
    return examples


def load_local_pdf_examples(raw_dir: Path) -> list[dict]:
    from scripts.build_risk_training_data import bootstrap_annotated_from_pdfs, build_training_examples

    annotated_path = Path("data/processed/_local_pdf_contracts.json")
    training_path = Path("data/processed/_local_rule_training.json")
    pdfs = list(Path(raw_dir).glob("*.pdf"))
    if not pdfs:
        return []

    bootstrap_annotated_from_pdfs(str(raw_dir), str(annotated_path))
    build_training_examples(str(annotated_path), str(training_path))
    with open(training_path, encoding="utf-8") as f:
        return json.load(f)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build risk training rows from public CUAD annotations"
    )
    parser.add_argument("--output", default="data/processed/risk_training_data.json")
    parser.add_argument("--cuad-dir", default="data/raw/cuad")
    parser.add_argument("--include-local-pdfs", action="store_true")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--excluded-md",
        default="",
        help="Optional markdown list of CUAD categories left unmapped on purpose.",
    )
    args = parser.parse_args()

    cuad_json = download_cuad(Path(args.cuad_dir))
    examples = span_examples_from_cuad(cuad_json)
    if args.excluded_md:
        counts = raw_category_span_counts(cuad_json)
        lines = [
            "# Categories left unmapped",
            "",
            "These CUAD categories are not assigned HIGH, MEDIUM, or LOW. Counts are raw answer spans.",
            "",
            "| Spans | Category |",
            "| --- | --- |",
        ]
        for name in EXCLUDED_CATEGORIES:
            lines.append(f"| {counts[name]} | {name} |")
        excluded_path = Path(args.excluded_md)
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
        print(f"Wrote {excluded_path}")

    if args.include_local_pdfs:
        examples.extend(load_local_pdf_examples(Path(args.raw_dir)))

    if len(examples) < 50:
        raise RuntimeError(
            f"Only {len(examples)} examples produced. CUAD download or parsing may have failed."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    counts = Counter(row["label"] for row in examples)
    print(f"Wrote {len(examples)} examples -> {args.output}")
    print(f"  contracts: {len({row['contract_id'] for row in examples if row.get('contract_id')})}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
