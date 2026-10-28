"""Shared ISOT loading and text preparation used by every model."""
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
        title = frame[title_col].fillna("").astype(str) if title_col else ""
        body = frame[body_col].fillna("").astype(str)
        text = (title + " [SEP] " + body).str.strip() if title_col else body.str.strip()
        part = pd.DataFrame({"label": label, "text": text})
        parts.append(part)

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
    return data[["id", "label", "text"]].reset_index(drop=True)
