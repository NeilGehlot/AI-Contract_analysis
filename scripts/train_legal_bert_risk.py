"""Fine-tune LEGAL-BERT for HIGH, MEDIUM, and LOW risk. Needs a CUDA GPU.

    Run from the repo root:
        python scripts/train_legal_bert_risk.py --out-dir data/models/legal_bert_risk --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.legal_bert_risk_classifier import LABEL2ID, ID2LABEL
from src.utils.cpu_thermal_guard import CpuThermalGuard
from src.utils.gpu_thermal_guard import GpuThermalGuard

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


class RiskDataset(Dataset):
    """Tokenize on the fly so batches pad to the longest clause, not 256."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item["labels"] = self.labels[idx]
        return item


def freeze_base_layers(model, unfreeze_last_n: int = 2) -> None:
    """On CPU, freeze embeddings + early encoder layers to cut backward cost."""
    if unfreeze_last_n < 0:
        return
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    encoder_layers = model.bert.encoder.layer
    freeze_until = max(0, len(encoder_layers) - unfreeze_last_n)
    for layer in encoder_layers[:freeze_until]:
        for param in layer.parameters():
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    report = classification_report(
        labels, preds, target_names=target_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "high_f1": report.get("HIGH", {}).get("f1-score", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LEGAL-BERT for risk classification")
    parser.add_argument(
        "--data", default="data/processed/risk_training_data.json"
    )
    parser.add_argument(
        "--output", default="data/models/risk_classifier_final"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all data")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--init-from",
        default="",
        help="Existing checkpoint to continue from. Empty = base LEGAL-BERT.",
    )
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=-1,
        help="Unfreeze last N BERT encoder layers; -1 = train all layers",
    )
    parser.add_argument("--train-json", default="", help="Optional train JSON. Skips the internal split.")
    parser.add_argument("--val-json", default="", help="Optional validation JSON. Requires --train-json.")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional model directory. Empty keeps --output and the default checkpoint folder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if bool(args.train_json) != bool(args.val_json):
        raise SystemExit("--train-json and --val-json must be passed together")

    if args.train_json:
        data_path = Path(args.train_json)
    else:
        data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run scripts/build_training_data_from_cuad.py first."
        )

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples > 0:
        data = data[: args.max_samples]

    texts, labels = [], []
    for row in data:
        label = row.get("label")
        if label not in LABEL2ID:
            continue
        texts.append(row["text"])
        labels.append(LABEL2ID[label])

    init_from = args.init_from.strip()
    tokenizer_source = init_from or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if init_from:
        model = AutoModelForSequenceClassification.from_pretrained(init_from)
        print(f"Continued from {init_from}")
    else:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.num_labels = 3
        config.id2label = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        config.label2id = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        config.problem_type = "single_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True,
        )
    print(f"Classifier labels: {model.config.num_labels} -> {model.config.id2label}")
    freeze_base_layers(model, args.unfreeze_last_n)
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError(
            "CUDA is not available. Install a GPU PyTorch wheel, e.g.\n"
            "  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126"
        )
    model.gradient_checkpointing_enable()
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if args.train_json:
        val_data = json.loads(Path(args.val_json).read_text(encoding="utf-8"))
        val_texts, val_labels = [], []
        for row in val_data:
            label = row.get("label")
            if label not in LABEL2ID:
                continue
            val_texts.append(row["text"])
            val_labels.append(LABEL2ID[label])
        train_texts, train_labels = texts, labels
    else:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.15, stratify=labels, random_state=42
        )

    train_ds = RiskDataset(train_texts, train_labels, tokenizer)
    val_ds = RiskDataset(val_texts, val_labels, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    model_dir = Path(args.out_dir) if args.out_dir else Path(args.output)
    checkpoint_dir = str(model_dir / "checkpoints") if args.out_dir else "data/models/risk_classifier_checkpoints"
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
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

    print(f"Training LEGAL-BERT on {len(train_texts)} examples ({args.epochs} epochs)...")
    print(
        f"Device: cuda  fp16=True  batch={args.batch_size}  "
        f"grad_accum={args.grad_accum}  effective_batch={args.batch_size * args.grad_accum}"
    )
    trainer.train()

    output_path = model_dir
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    trainer.state.save_to_json(str(output_path / "trainer_state.json"))

    predictions = trainer.predict(val_ds)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    report_text = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    print("\n=== LEGAL-BERT risk validation ===")
    print(report_text)

    if args.out_dir:
        metrics_dir = output_path
    else:
        metrics_dir = ROOT / "outputs" / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "legal_bert_risk_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_size": len(train_texts),
                "val_size": len(val_texts),
                "epochs": args.epochs,
                "trainer_metrics": predictions.metrics,
                "classification_report": report_dict,
            },
            f,
            indent=2,
        )
    md_path = metrics_dir / "legal_bert_risk_metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# LEGAL-BERT risk classifier (validation)\n\n")
        f.write(f"- Train examples: **{len(train_texts)}**\n")
        f.write(f"- Val examples: **{len(val_texts)}**\n")
        f.write(f"- Epochs: **{args.epochs}**\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("```\n")
    print(f"Saved model to {output_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
