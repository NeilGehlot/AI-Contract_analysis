"""Fine-tune LEGAL-BERT for contract named-entity recognition. Needs a CUDA GPU.

    Run from the repo root:
        python scripts/train_legal_bert_ner.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.cpu_thermal_guard import CpuThermalGuard
from src.utils.gpu_thermal_guard import GpuThermalGuard

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

ENTITY_LABELS = ["PARTY", "DATE", "AMOUNT", "OBLIGATION", "TERM", "CLAUSE"]
LABEL_LIST = ["O"] + [f"B-{l}" for l in ENTITY_LABELS] + [f"I-{l}" for l in ENTITY_LABELS]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def spans_to_bio_labels(text: str, entities: list[dict], offsets: list[tuple]) -> list[int]:
    """Align character spans to token BIO labels using offset mapping."""
    labels = [LABEL2ID["O"]] * len(offsets)

    for ent in entities:
        ent_label = ent["label"]
        start, end = ent["start"], ent["end"]
        first = True
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start is None or tok_end is None:
                continue
            if tok_end <= start or tok_start >= end:
                continue
            prefix = "B" if first else "I"
            labels[idx] = LABEL2ID[f"{prefix}-{ent_label}"]
            first = False

    return labels


class NERDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.encodings = []
        self.labels = []

        for example in examples:
            encoding = tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_offsets_mapping=True,
            )
            offsets = encoding.pop("offset_mapping")
            token_labels = spans_to_bio_labels(example["text"], example["entities"], offsets)
            # Replace special token positions with -100
            aligned = []
            for i, (start, _) in enumerate(offsets):
                if start is None:
                    aligned.append(-100)
                else:
                    aligned.append(token_labels[i])

            self.encodings.append(encoding)
            self.labels.append(aligned)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []
    for pred_row, label_row in zip(predictions, labels):
        true_seq, pred_seq = [], []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue
            true_seq.append(ID2LABEL[label_id])
            pred_seq.append(ID2LABEL[pred_id])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LEGAL-BERT for NER")
    parser.add_argument("--data", default="data/processed/ner_training_data.json")
    parser.add_argument("--output", default="data/models/ner")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run scripts/build_ner_training_data_from_cuad.py first."
        )

    with open(data_path, encoding="utf-8") as f:
        examples = json.load(f)

    if args.max_samples > 0:
        examples = examples[: args.max_samples]

    train_ex, val_ex = train_test_split(examples, test_size=0.15, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a GPU PyTorch wheel, e.g.\n"
            "  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126"
        )
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.gradient_checkpointing_enable()
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    train_ds = NERDataset(train_ex, tokenizer)
    val_ds = NERDataset(val_ex, tokenizer)
    collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir="data/models/ner_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        save_total_limit=2,
        dataloader_pin_memory=True,
        fp16=True,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[GpuThermalGuard(), CpuThermalGuard()],
    )

    print(f"Training LEGAL-BERT NER on {len(train_ex)} examples ({args.epochs} epochs)...")
    print(
        f"Device: cuda  fp16=True  batch={args.batch_size}  "
        f"grad_accum={args.grad_accum}  effective_batch={args.batch_size * args.grad_accum}"
    )
    trainer.train()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Save label map for inference
    with open(output_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    print(f"Saved NER model to {output_path}")


if __name__ == "__main__":
    main()
