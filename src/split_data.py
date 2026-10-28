"""Create deterministic ISOT splits and a shareable article-text-free manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_isot


SPLIT_NAMES = ("train", "validation", "test")


def _assign_from_manifest(data: pd.DataFrame, manifest_path: str | Path) -> dict[str, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, dtype={"id": "string"})
    manifest["id"] = manifest["id"].astype(str).str.strip()
    manifest["label"] = pd.to_numeric(manifest["label"], errors="coerce")
    required = {"id", "label", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    if manifest["id"].eq("").any() or manifest["id"].duplicated().any():
        raise ValueError("Manifest contains empty or duplicate article IDs")
    if manifest["label"].isna().any() or not manifest["label"].isin([0, 1]).all():
        raise ValueError("Manifest labels must be 0 (fake) or 1 (real)")
    if not manifest["split"].isin(SPLIT_NAMES).all():
        raise ValueError(f"Manifest split values must be one of {SPLIT_NAMES}")

    current = data[["id", "label"]].copy()
    current["id"] = current["id"].astype(str).str.strip()
    current["label"] = pd.to_numeric(current["label"], errors="raise")
    manifest = manifest[["id", "label", "split"]].copy()
    merged = current.merge(
        manifest, on="id", how="outer", suffixes=("_current", "_manifest"), indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Manifest article IDs do not match the current dataset")
    if not merged["label_current"].eq(merged["label_manifest"]).all():
        raise ValueError("Manifest labels do not match the current dataset")

    assignments = manifest.set_index("id")["split"]
    result = data.copy()
    result["split"] = result["id"].map(assignments)
    if result["split"].isna().any():
        raise ValueError("Manifest did not assign every article")
    return {name: result[result["split"].eq(name)].drop(columns="split").copy() for name in SPLIT_NAMES}


def _make_manifest(data: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        frame[["id", "label"]].assign(split=name)
        for name, frame in splits.items()
    ]
    return pd.concat(rows, ignore_index=True).sort_values("id").reset_index(drop=True)


def create_splits(
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    manifest_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    data = load_isot(data_dir).sort_values("id").reset_index(drop=True)
    if data["label"].nunique() != 2:
        raise ValueError("The dataset must contain both fake (0) and real (1) examples.")
    counts = data["label"].value_counts()
    if int(counts.min()) < 4:
        raise ValueError("Need at least 4 examples in each class to create stratified splits.")

    if manifest_path:
        splits = _assign_from_manifest(data, manifest_path)
    else:
        train, remainder = train_test_split(
            data, test_size=0.30, random_state=seed, stratify=data["label"]
        )
        validation, test = train_test_split(
            remainder,
            test_size=0.50,
            random_state=seed,
            stratify=remainder["label"],
        )
        splits = {"train": train, "validation": validation, "test": test}

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.sort_values("id").to_csv(output / f"{name}.csv", index=False)

    manifest = _make_manifest(data, splits)
    manifest.to_csv(output / "split_manifest.csv", index=False)
    dataset_fingerprint = hashlib.sha256(
        "\n".join(
            f"{row.id}:{row.label}"
            for row in data[["id", "label"]].itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "seed": seed if manifest_path is None else None,
        "assignment_source": str(manifest_path) if manifest_path else "stratified random split",
        "ratios": {name: len(frame) / len(data) for name, frame in splits.items()},
        "rows": {name: len(frame) for name, frame in splits.items()},
        "dataset_fingerprint_sha256": dataset_fingerprint,
    }
    (output / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Saved {len(splits['train'])} train, {len(splits['validation'])} validation, "
        f"and {len(splits['test'])} test rows to {output}."
    )
    print(f"Article-text-free assignment manifest: {output / 'split_manifest.csv'}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Reuse an ID-only split_manifest.csv created by an earlier run.",
    )
    args = parser.parse_args()
    create_splits(args.data_dir, args.output_dir, args.seed, args.manifest)


if __name__ == "__main__":
    main()
