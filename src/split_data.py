"""Create deterministic, stratified ISOT train/validation/test CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data import load_isot


def create_splits(data_dir: str, output_dir: str, seed: int = 42) -> None:
    data = load_isot(data_dir)
    train, remainder = train_test_split(
        data, test_size=0.30, random_state=seed, stratify=data["label"]
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder["label"],
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        frame.sort_values("id").to_csv(output / f"{name}.csv", index=False)

    print(
        f"Saved {len(train)} train, {len(validation)} validation, "
        f"and {len(test)} test rows to {output} (seed={seed})."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_splits(args.data_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
