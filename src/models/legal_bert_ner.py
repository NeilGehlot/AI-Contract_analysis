"""LEGAL-BERT NER entity extractor."""

import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ENTITY_GROUPS = {
    "PARTY": "parties",
    "DATE": "dates",
    "AMOUNT": "amounts",
    "OBLIGATION": "obligations",
    "TERM": "terms",
    "CLAUSE": "clauses",
}


class LegalBertEntityExtractor:
    """Extract entities using fine-tuned LEGAL-BERT token classifier."""

    def __init__(self, model_path: str = "data/models/ner"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        label_map_path = Path(model_path) / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as f:
                maps = json.load(f)
            self.id2label = {int(k): v for k, v in maps["id2label"].items()}
        else:
            self.id2label = self.model.config.id2label

        logger.info("LegalBertEntityExtractor initialized")

    @classmethod
    def is_available(cls, model_path: str = "data/models/ner") -> bool:
        path = Path(model_path)
        return path.exists() and (path / "config.json").exists()

    def extract_all(self, text: str) -> Dict:
        logger.info("Extracting entities with LEGAL-BERT NER...")
        ner_entities = self._extract_ner_entities(text)

        return {
            "parties": ner_entities.get("parties", [])[:10],
            "dates": ner_entities.get("dates", [])[:10],
            "amounts": self._extract_amounts_regex(text),
            "emails": self._extract_emails(text),
            "terms": self._terms_from_ner(ner_entities),
            "obligations": ner_entities.get("obligations", [])[:10],
            "clauses": ner_entities.get("clauses", [])[:10],
        }

    @torch.no_grad()
    def _extract_ner_entities(self, text: str) -> Dict[str, List]:
        chunk_size = 450
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - 50)]
        grouped: Dict[str, List[str]] = {k: [] for k in ENTITY_GROUPS.values()}

        for chunk in chunks:
            encoding = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )
            offsets = encoding.pop("offset_mapping")[0].tolist()
            inputs = {k: v.to(self.device) for k, v in encoding.items()}
            logits = self.model(**inputs).logits[0]
            pred_ids = torch.argmax(logits, dim=-1).tolist()

            current_label = None
            current_tokens = []

            for pred_id, (start, end) in zip(pred_ids, offsets):
                if start is None:
                    continue
                label = self.id2label.get(pred_id, self.id2label.get(str(pred_id), "O"))
                if label == "O":
                    if current_label and current_tokens:
                        self._append_entity(grouped, current_label, chunk, current_tokens)
                    current_label = None
                    current_tokens = []
                    continue

                prefix, ent_type = label.split("-", 1)
                if prefix == "B" or current_label != ent_type:
                    if current_label and current_tokens:
                        self._append_entity(grouped, current_label, chunk, current_tokens)
                    current_label = ent_type
                    current_tokens = [(start, end)]
                else:
                    current_tokens.append((start, end))

            if current_label and current_tokens:
                self._append_entity(grouped, current_label, chunk, current_tokens)

        for key in grouped:
            grouped[key] = list(dict.fromkeys(grouped[key]))
        return grouped

    def _append_entity(
        self, grouped: Dict[str, List], ent_type: str, text: str, spans: list
    ):
        key = ENTITY_GROUPS.get(ent_type)
        if not key:
            return
        start = spans[0][0]
        end = spans[-1][1]
        value = text[start:end].strip()
        if len(value) > 2:
            grouped[key].append(value)

    def _terms_from_ner(self, ner_entities: Dict) -> Dict:
        terms = {}
        term_spans = ner_entities.get("terms", [])
        if term_spans:
            terms["extracted_terms"] = "; ".join(term_spans[:5])
        return terms

    def _extract_amounts_regex(self, text: str) -> List[Dict]:
        amounts = []
        patterns = [
            (r"₹\s*(\d+(?:,\d+)*(?:\.\d+)?)", "₹"),
            (r"Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)", "₹"),
            (r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)", "$"),
        ]
        for pattern, currency in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amount_str = match.group(1).replace(",", "")
                try:
                    amounts.append(
                        {
                            "amount": match.group(1),
                            "currency": currency,
                            "value": float(amount_str),
                        }
                    )
                except ValueError:
                    pass
        amounts.sort(key=lambda x: x["value"], reverse=True)
        return amounts[:10]

    def _extract_emails(self, text: str) -> List[str]:
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        return list(set(re.findall(pattern, text)))[:5]
