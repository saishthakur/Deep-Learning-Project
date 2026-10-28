# Dataset setup

Place the ISOT dataset files in this folder:

- `Fake.csv`
- `True.csv`

The loader expects each file to contain a `text` column and uses the `title` column when present. It assigns fake = `0` and real = `1`, removes empty and normalized duplicate articles, and stops if identical normalized text has conflicting labels so it can be reviewed.

Run the split command from the repository root. It writes the model-ready CSVs, an article-text-free `split_manifest.csv`, and metadata under `data/processed/`:

```powershell
python -m src.split_data --data-dir data --output-dir data/processed --seed 42
```

Share `split_manifest.csv` with teammates to reproduce the same assignment without sharing article text. The manifest contains hashed IDs, labels, and split names, but no article text. To apply a shared manifest:

```powershell
python -m src.split_data --data-dir data --output-dir data/processed --manifest path\to\split_manifest.csv
```

The generated split CSVs contain article text, so keep them and the original dataset files out of commits. Confirm the dataset's source and usage terms with your course/project instructions.
