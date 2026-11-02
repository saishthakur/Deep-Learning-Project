"""Shared ISOT loading, preprocessing, split validation, and deduplication."""
from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import pandas as pd


def clean_article(text: str) -> str:
    """Remove common publisher/markup shortcuts and normalize article text."""
    value = html.unescape(str(text))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"https?://\S+|www\.\S+", " ", value, flags=re.IGNORECASE)
    title_body = re.split(r"\s+\[sep\]\s+", value, maxsplit=1, flags=re.IGNORECASE)
    def strip_reuters_prefix(body: str) -> str:
        for _ in range(5):
            body, removed = re.subn(
                r"^[\s\S]{0,1000}?\(reuters\)\s*[-–—]\s*", "", body, count=1, flags=re.IGNORECASE
            )
            if not removed:
                break
        return body

    if len(title_body) == 2:
        title, body = title_body
        body = strip_reuters_prefix(body)
        value = f"{title} [SEP] {body}"
    else:
        value = strip_reuters_prefix(value)
    return re.sub(r"\s+", " ", value).strip().casefold().replace("[sep]", "[SEP]")


def _normalise_for_dedup(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def load_isot(data_dir: str | Path) -> pd.DataFrame:
    """Load Fake.csv and True.csv; return unique rows with id, label, and text."""
    data_dir = Path(data_dir)
    parts = []
    for filename, label in (("Fake.csv", 0), ("True.csv", 1)):
        path = data_dir / filename
        if not path.is_file(): raise FileNotFoundError(f"Expected dataset file: {path}")
        frame = pd.read_csv(path)
        columns = {str(column).strip().casefold(): column for column in frame.columns}
        if "text" not in columns: raise ValueError(f"{path} must contain a 'text' column")
        title_col, body_col = columns.get("title"), columns["text"]
        body = frame[body_col].fillna("").astype(str)
        if title_col:
            title = frame[title_col].fillna("").astype(str)
            raw = (title + " [SEP] " + body).str.strip()
        else: raw = body.str.strip()
        parts.append(pd.DataFrame({"label": label, "text": raw.map(clean_article)}))
    data = pd.concat(parts, ignore_index=True)
    data = data[data.text.ne("")].copy()
    data["_dedup"] = data.text.map(_normalise_for_dedup)
    conflicting = data.groupby("_dedup").label.nunique()
    if (conflicting > 1).any():
        count = int((conflicting > 1).sum())
        raise ValueError(f"Found {count} identical normalized article(s) with conflicting labels. Review these rows before splitting.")
    data = data.drop_duplicates("_dedup", keep="first").drop(columns="_dedup")
    data["id"] = data.text.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:16])
    if data.id.duplicated().any(): raise ValueError("Article ID collision detected; review the source data.")
    return data[["id", "label", "text"]].reset_index(drop=True)


def validate_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    required = {"id", "label", "text"}; missing = required.difference(frame.columns)
    if missing: raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")
    if frame.empty: raise ValueError(f"{split_name} split has no rows")
    result = frame[["id", "label", "text"]].copy()
    result.id = result.id.astype("string").str.strip()
    result.text = result.text.fillna("").astype(str).str.strip()
    labels = pd.to_numeric(result.label, errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all(): raise ValueError(f"{split_name} labels must be 0 (fake) or 1 (real)")
    if result.id.isna().any() or result.id.eq("").any(): raise ValueError(f"{split_name} contains an empty article ID")
    if result.id.duplicated().any(): raise ValueError(f"{split_name} contains duplicate article IDs")
    if result.text.eq("").any(): raise ValueError(f"{split_name} contains empty article text")
    result.label = labels.astype(int)
    return result.reset_index(drop=True)


def load_split(path: str | Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": "string"})
    return validate_split(frame, split_name)

