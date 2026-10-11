"""Build NER training examples from CUAD expert span annotations.

    Run from the repo root:
        python scripts/build_ner_training_data_from_cuad.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.cuad_utils import download_cuad_json, normalize_question

def category_key(name: str) -> str:
    """Lowercase and collapse whitespace so titles match despite case and spacing."""
    return " ".join(name.strip().lower().split())


# Real CUAD question titles. Lookup uses category_key.
CUAD_TO_ENTITY = {
    "Parties": "PARTY",
    "Agreement Date": "DATE",
    "Effective Date": "DATE",
    "Expiration Date": "DATE",
    "Renewal Term": "TERM",
    "Notice Period To Terminate Renewal": "TERM",
    "Minimum Commitment": "OBLIGATION",
    "Post-Termination Services": "OBLIGATION",
    "Revenue/Profit Sharing": "OBLIGATION",
    "Audit Rights": "OBLIGATION",
    "Uncapped Liability": "CLAUSE",
    "Cap On Liability": "CLAUSE",
    "Liquidated Damages": "CLAUSE",
    "Termination For Convenience": "CLAUSE",
    "Governing Law": "TERM",
    "License Grant": "OBLIGATION",
    "Ip Ownership Assignment": "OBLIGATION",
}
ENTITY_BY_KEY = {category_key(name): label for name, label in CUAD_TO_ENTITY.items()}


def chunk_text_with_entities(
    text: str, entities: list[dict], window: int = 1500, stride: int = 750
) -> list[dict]:
    if len(text) <= window:
        return [{"text": text, "entities": entities}]

    chunks = []
    for start in range(0, len(text), stride):
        end = min(start + window, len(text))
        chunk_text = text[start:end]
        chunk_entities = []
        for ent in entities:
            if ent["start"] >= start and ent["end"] <= end:
                chunk_entities.append(
                    {
                        "start": ent["start"] - start,
                        "end": ent["end"] - start,
                        "label": ent["label"],
                    }
                )
        if chunk_entities:
            chunks.append({"text": chunk_text, "entities": chunk_entities})
        if end >= len(text):
            break
    return chunks


def build_ner_examples(cuad_json_path: Path, max_text_length: int = 50000) -> list[dict]:
    with open(cuad_json_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    seen_keys = set()

    for item in data["data"]:
        for paragraph in item.get("paragraphs", []):
            text = paragraph.get("context", "").strip()
            if len(text) < 30 or len(text) > max_text_length:
                continue

            entities = []
            for qa in paragraph.get("qas", []):
                category = normalize_question(qa.get("question", ""))
                seen_keys.add(category_key(category))
                entity_type = ENTITY_BY_KEY.get(category_key(category))
                if not entity_type:
                    continue

                for answer in qa.get("answers") or []:
                    answer_text = (answer.get("text") or "").strip()
                    start = answer.get("answer_start", -1)
                    if start < 0 or len(answer_text) < 2:
                        continue
                    end = start + len(answer_text)
                    if end > len(text):
                        continue
                    entities.append(
                        {"start": start, "end": end, "label": entity_type}
                    )

            if not entities:
                continue

            for chunk in chunk_text_with_entities(text, entities):
                examples.append(chunk)

    missing = [name for name in CUAD_TO_ENTITY if category_key(name) not in seen_keys]
    if missing:
        print("Mapped names with no raw CUAD match:")
        for name in missing:
            print(f"  {name}")
    else:
        print("Every mapped NER name matched a raw CUAD category.")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Build NER training data from CUAD")
    parser.add_argument("--cuad-dir", default="data/raw/cuad")
    parser.add_argument("--output", default="data/processed/ner_training_data.json")
    args = parser.parse_args()

    cuad_json = download_cuad_json(args.cuad_dir)
    examples = build_ner_examples(cuad_json)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    label_counts = Counter()
    for ex in examples:
        for ent in ex["entities"]:
            label_counts[ent["label"]] += 1

    print(f"Wrote {len(examples)} examples -> {args.output}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} spans")


if __name__ == "__main__":
    main()
