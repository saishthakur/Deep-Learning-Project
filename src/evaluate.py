"""Evaluate a saved classifier on one prepared split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.train_transformer import NewsDataset, make_collator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/distilbert")
    parser.add_argument("--split", default="data/processed/test.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    frame = pd.read_csv(args.split)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Split CSV must contain text and label columns")
    config_path = model_dir / "run_config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    max_length = args.max_length or saved.get("max_length", 256)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(
        NewsDataset(frame), batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collator(tokenizer, max_length)
    )

    actual, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch).logits.argmax(dim=1).cpu()
            actual.extend(labels.tolist())
            predicted.extend(outputs.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", pos_label=1, zero_division=0
    )
    results = {
        "split": str(args.split),
        "examples": len(actual),
        "positive_label": "real (1)",
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_real": float(precision),
        "recall_real": float(recall),
        "f1_real": float(f1),
        "confusion_matrix_labels_fake0_real1": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            actual, predicted, labels=[0, 1], target_names=["fake", "real"],
            output_dict=True, zero_division=0
        ),
    }
    print(json.dumps(results, indent=2))
    results_path = model_dir / "test_metrics.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics to {results_path}")


if __name__ == "__main__":
    main()
