"""Fine-tune a DistilBERT text classifier on the shared ISOT split."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NewsDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {"text": self.texts[index], "label": self.labels[index]}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collator(tokenizer, max_length: int):
    def collate(rows):
        encoded = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([row["label"] for row in rows])
        return encoded
    return collate


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    actual, predicted = [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--validation", default="data/processed/validation.csv")
    parser.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--output-dir", default="artifacts/distilbert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        parser.error("epochs/batch-size must be positive and max-length must be at least 8")

    seed_everything(args.seed)
    train_frame = pd.read_csv(args.train)
    validation_frame = pd.read_csv(args.validation)
    for name, frame in (("train", train_frame), ("validation", validation_frame)):
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"{name} CSV must contain text and label columns")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = make_collator(tokenizer, args.max_length)
    train_loader = DataLoader(
        NewsDataset(train_frame), batch_size=args.batch_size, shuffle=True,
        collate_fn=collator
    )
    validation_loader = DataLoader(
        NewsDataset(validation_frame), batch_size=args.batch_size, shuffle=False,
        collate_fn=collator
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += result.loss.item()

        metrics = score(model, validation_loader, device)
        metrics.update(epoch=epoch, train_loss=total_loss / max(len(train_loader), 1))
        history.append(metrics)
        print(json.dumps(metrics))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            (output_dir / "run_config.json").write_text(
                json.dumps({
                    "base_model": args.model_name,
                    "seed": args.seed,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "epochs_requested": args.epochs,
                    "best_validation_f1": best_f1,
                    "label_mapping": {"fake": 0, "real": 1},
                    "history": history,
                }, indent=2),
                encoding="utf-8",
            )

    print(f"Best validation F1: {best_f1:.4f}; saved model to {output_dir}")


if __name__ == "__main__":
    main()
