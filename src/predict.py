"""Predict fake/real for a supplied title and article body."""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/distilbert")
    parser.add_argument("--title", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    combined = f"{args.title} [SEP] {args.text}" if args.title else args.text
    inputs = tokenizer(
        combined, return_tensors="pt", truncation=True,
        max_length=args.max_length
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)[0]
    predicted = int(probabilities.argmax().item())
    print({
        "label": "real" if predicted == 1 else "fake",
        "label_id": predicted,
        "probabilities": {
            "fake": round(float(probabilities[0].item()), 6),
            "real": round(float(probabilities[1].item()), 6),
        },
        "warning": "Model prediction from learned dataset patterns; not claim verification.",
    })


if __name__ == "__main__":
    main()
