"""Train one model in the shared MLP/CNN/LSTM/DistilBERT experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data import load_split
from src.neural_models import TextCNN, TextLSTM, build_vocabulary, encode_text, parameter_count


class SparseDataset(Dataset):
    def __init__(self, matrix, labels): self.matrix, self.labels = matrix, labels.astype("float32")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return torch.from_numpy(self.matrix[i].toarray().ravel()), torch.tensor(self.labels[i])


class SequenceDataset(Dataset):
    def __init__(self, frame, vocab):
        self.ids = [encode_text(text, vocab) for text in frame.text]
        self.labels = frame.label.to_numpy(dtype="float32")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return torch.tensor(self.ids[i]), torch.tensor(self.labels[i])


class TransformerDataset(Dataset):
    def __init__(self, frame): self.texts, self.labels = frame.text.tolist(), frame.label.astype(int).tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.texts[i], self.labels[i]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("mlp", "cnn", "lstm", "distilbert"), required=True)
    p.add_argument("--train", default="data/processed/train.csv")
    p.add_argument("--validation", default="data/processed/validation.csv")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=None)
    a = p.parse_args()
    is_bert = a.model == "distilbert"
    epochs = a.epochs or (3 if is_bert else 20)
    batch_size = a.batch_size or (16 if is_bert else 64)
    lr = a.learning_rate or (2e-5 if is_bert else 1e-3)
    max_length = a.max_length or (256 if is_bert else 400)
    if min(epochs, batch_size, a.patience, max_length) < 1 or lr <= 0: p.error("training settings must be positive")
    seed_all(a.seed)
    train, valid = load_split(a.train, "train"), load_split(a.validation, "validation")
    if set(train.id) & set(valid.id): raise ValueError("Train and validation article IDs overlap")
    output = Path(a.output_dir or f"artifacts/{a.model}"); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = None; vectorizer = None; tokenizer = None; scheduler = None

    if a.model == "mlp":
        vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), dtype=np.float32, sublinear_tf=True)
        x_train = vectorizer.fit_transform(train.text); x_valid = vectorizer.transform(valid.text)
        joblib.dump(vectorizer, output / "vectorizer.joblib")
        model = nn.Sequential(nn.Linear(x_train.shape[1], 256), nn.ReLU(), nn.Dropout(.5), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 1)).to(device)
        train_loader = DataLoader(SparseDataset(x_train, train.label), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(SparseDataset(x_valid, valid.label), batch_size=batch_size)
        def logits_for(batch): return model(batch[0].to(device))
        def labels_for(batch): return batch[1].to(device)
    elif a.model in ("cnn", "lstm"):
        vocab = build_vocabulary(train.text, 30_000)
        with (output / "vocabulary.json").open("w", encoding="utf-8") as f: json.dump(vocab, f)
        model = (TextCNN if a.model == "cnn" else TextLSTM)(len(vocab) + 2).to(device)
        train_loader = DataLoader(SequenceDataset(train, vocab), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(SequenceDataset(valid, vocab), batch_size=batch_size)
        def logits_for(batch): return model(batch[0].to(device))
        def labels_for(batch): return batch[1].to(device)
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
        tokenizer = AutoTokenizer.from_pretrained(a.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(a.model_name, num_labels=2, id2label={0:"fake",1:"real"}, label2id={"fake":0,"real":1}).to(device)
        def collate(rows):
            inputs = tokenizer([r[0] for r in rows], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            inputs["labels"] = torch.tensor([r[1] for r in rows]); return inputs
        train_loader = DataLoader(TransformerDataset(train), batch_size=batch_size, shuffle=True, collate_fn=collate)
        valid_loader = DataLoader(TransformerDataset(valid), batch_size=batch_size, collate_fn=collate)
        scheduler = get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01 if is_bert else 0.0)
    if is_bert: scheduler = scheduler(optimizer, int(.1 * epochs * len(train_loader)), epochs * len(train_loader))
    criterion = nn.BCEWithLogitsLoss()
    best_loss, bad_epochs, history = float("inf"), 0, []
    best_path = output / "best_model.pt"
    for epoch in range(1, epochs + 1):
        start = time.perf_counter(); model.train(); losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if is_bert:
                batch = {k:v.to(device) for k,v in batch.items()}
                result = model(**batch); loss = result.loss; scores = result.logits[:, 1] - result.logits[:, 0]
                labels = batch["labels"].float()
            else:
                scores, labels = logits_for(batch), labels_for(batch); loss = criterion(scores, labels)
            loss.backward()
            if is_bert or a.model == "lstm": nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None: scheduler.step()
            losses.append(float(loss.detach().cpu()))
        train_seconds = time.perf_counter() - start
        model.eval(); val_losses, ys, ps = [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                if is_bert:
                    batch = {k:v.to(device) for k,v in batch.items()}; labels = batch.pop("labels")
                    raw = model(**batch).logits; loss = nn.functional.cross_entropy(raw, labels)
                    scores = raw[:, 1] - raw[:, 0]; labels = labels.float()
                else:
                    scores, labels = logits_for(batch), labels_for(batch); loss = criterion(scores, labels)
                val_losses.append(float(loss.cpu())); ys.extend(labels.cpu().numpy().astype(int)); ps.extend((scores.cpu().numpy() >= 0).astype(int))
        val_loss = float(np.mean(val_losses)); f1_fake = float(f1_score(ys, ps, pos_label=0, zero_division=0))
        record = {"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,"validation_f1_fake":f1_fake,"train_seconds":train_seconds}
        history.append(record); print(json.dumps(record))
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            if is_bert:
                model.save_pretrained(output / "transformer"); tokenizer.save_pretrained(output / "transformer")
            else: torch.save(model.state_dict(), best_path)
            train_ids=sorted(train.id.astype(str)); validation_ids=sorted(valid.id.astype(str))
            config = {"model":a.model,"model_name":a.model_name if is_bert else None,"seed":a.seed,"max_length":max_length,"batch_size":batch_size,"learning_rate":lr,"epochs_run":epoch,"patience":a.patience,"best_validation_loss":best_loss,"best_validation_f1_fake":f1_fake,"parameter_count":parameter_count(model),"train_ids":train_ids,"validation_ids":validation_ids,"train_ids_sha256":hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),"validation_ids_sha256":hashlib.sha256("\n".join(validation_ids).encode()).hexdigest(),"history":history,"device":str(device)}
            (output / "run_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience: print(f"Early stopping at epoch {epoch}"); break
    print(f"Best validation loss: {best_loss:.6f}; artifacts: {output}")


if __name__ == "__main__": main()

