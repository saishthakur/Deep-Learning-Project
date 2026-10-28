"""Shared ISOT loading, split validation, and text preparation."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


def _normalise_for_dedup(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def load_isot(data_dir: str | Path) -> pd.DataFrame:
    """Load Fake.csv and True.csv; return unique rows with id, label, and text."""
    data_dir = Path(data_dir)
    parts = []
    for filename, label in (("Fake.csv", 0), ("True.csv", 1)):
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected dataset file: {path}")
        frame = pd.read_csv(path)
        columns = {str(column).strip().casefold(): column for column in frame.columns}
        if "text" not in columns:
            raise ValueError(f"{path} must contain a 'text' column")
        title_col = columns.get("title")
        body_col = columns["text"]
        body = frame[body_col].fillna("").astype(str)
        if title_col:
            title = frame[title_col].fillna("").astype(str)
            text = (title + " [SEP] " + body).str.strip()
        else:
            text = body.str.strip()
        parts.append(pd.DataFrame({"label": label, "text": text}))

    data = pd.concat(parts, ignore_index=True)
    data["text"] = data["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    data = data[data["text"].ne("")].copy()
    data["_dedup"] = data["text"].map(_normalise_for_dedup)

    conflicting = data.groupby("_dedup")["label"].nunique()
    if (conflicting > 1).any():
        count = int((conflicting > 1).sum())
        raise ValueError(
            f"Found {count} identical normalized article(s) with conflicting labels. "
            "Review these rows before splitting to avoid label leakage."
        )

    data = data.drop_duplicates("_dedup", keep="first").drop(columns="_dedup")
    data["id"] = data["text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    )
    if data["id"].duplicated().any():
        raise ValueError("Article ID collision detected; review the source data.")
    return data[["id", "label", "text"]].reset_index(drop=True)


def validate_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Validate a prepared split and return normalized text/label/id columns."""
    required = {"id", "label", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{split_name} split has no rows")

    result = frame[["id", "label", "text"]].copy()
    result["id"] = result["id"].astype("string").str.strip()
    result["text"] = result["text"].fillna("").astype(str).str.strip()
    labels = pd.to_numeric(result["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{split_name} labels must be 0 (fake) or 1 (real)")
    if result["id"].isna().any() or result["id"].eq("").any():
        raise ValueError(f"{split_name} contains an empty article ID")
    if result["id"].duplicated().any():
        raise ValueError(f"{split_name} contains duplicate article IDs")
    if result["text"].eq("").any():
        raise ValueError(f"{split_name} contains empty article text")

    result["label"] = labels.astype(int)
    return result.reset_index(drop=True)


def load_split(path: str | Path, split_name: str) -> pd.DataFrame:
    """Read and validate a generated split while preserving string IDs."""
    frame = pd.read_csv(path, dtype={"id": "string"})
    return validate_split(frame, split_name)
