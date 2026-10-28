"""Small synthetic-data checks for the shared ISOT preparation pipeline."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import load_isot, validate_split
from src.split_data import create_splits


def write_dataset(root: Path, reverse: bool = False) -> None:
    fake = pd.DataFrame({
        "title": [f"Fake headline {index}" for index in range(20)],
        "text": [f"Fabricated report number {index}" for index in range(20)],
    })
    real = pd.DataFrame({
        "title": [f"Real headline {index}" for index in range(20)],
        "text": [f"Verified report number {index}" for index in range(20)],
    })
    if reverse:
        fake = fake.iloc[::-1]
        real = real.iloc[::-1]
    fake.to_csv(root / "Fake.csv", index=False)
    real.to_csv(root / "True.csv", index=False)


class DataLoaderTests(unittest.TestCase):
    def test_labels_title_and_body_are_standardized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_dataset(root)
            frame = load_isot(root)
            self.assertEqual(set(frame["label"]), {0, 1})
            self.assertTrue(frame["text"].str.contains(r"\[SEP\]").all())
            self.assertEqual(frame["id"].nunique(), len(frame))

    def test_conflicting_duplicate_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({"title": ["Same"], "text": ["Same article."]}).to_csv(
                root / "Fake.csv", index=False
            )
            pd.DataFrame({"title": ["Same"], "text": ["Same article!"]}).to_csv(
                root / "True.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                load_isot(root)

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({"headline": ["x"]}).to_csv(root / "Fake.csv", index=False)
            pd.DataFrame({"text": ["y"]}).to_csv(root / "True.csv", index=False)
            with self.assertRaisesRegex(ValueError, "text"):
                load_isot(root)

    def test_split_rejects_invalid_labels(self):
        frame = pd.DataFrame({
            "id": ["abc"], "label": [2], "text": ["a story"]
        })
        with self.assertRaisesRegex(ValueError, "0 \(fake\) or 1 \(real\)"):
            validate_split(frame, "test")


class ReproducibleSplitTests(unittest.TestCase):
    def test_order_independent_and_manifest_reusable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_data, second_data = root / "first", root / "second"
            first_data.mkdir()
            second_data.mkdir()
            write_dataset(first_data)
            write_dataset(second_data, reverse=True)

            first_out, second_out, reused_out = (
                root / "first_out", root / "second_out", root / "reused_out"
            )
            create_splits(first_data, first_out, seed=17)
            create_splits(second_data, second_out, seed=17)

            first_manifest = pd.read_csv(first_out / "split_manifest.csv", dtype={"id": str})
            second_manifest = pd.read_csv(second_out / "split_manifest.csv", dtype={"id": str})
            pd.testing.assert_frame_equal(first_manifest, second_manifest)

            create_splits(
                second_data,
                reused_out,
                seed=999,
                manifest_path=first_out / "split_manifest.csv",
            )
            reused_manifest = pd.read_csv(reused_out / "split_manifest.csv", dtype={"id": str})
            pd.testing.assert_frame_equal(first_manifest, reused_manifest)

            train_ids = set(pd.read_csv(reused_out / "train.csv", dtype={"id": str})["id"])
            validation_ids = set(pd.read_csv(reused_out / "validation.csv", dtype={"id": str})["id"])
            test_ids = set(pd.read_csv(reused_out / "test.csv", dtype={"id": str})["id"])
            self.assertFalse(train_ids & validation_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(validation_ids & test_ids)
            self.assertEqual(len(train_ids | validation_ids | test_ids), 40)

    def test_manifest_from_different_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            output = root / "output"
            data_dir.mkdir()
            write_dataset(data_dir)
            create_splits(data_dir, output, seed=17)

            manifest = pd.read_csv(output / "split_manifest.csv", dtype={"id": str})
            manifest.loc[0, "id"] = "not-a-current-article"
            bad_manifest = root / "bad_manifest.csv"
            manifest.to_csv(bad_manifest, index=False)

            with self.assertRaisesRegex(ValueError, "do not match"):
                create_splits(data_dir, root / "rejected", manifest_path=bad_manifest)


if __name__ == "__main__":
    unittest.main()
