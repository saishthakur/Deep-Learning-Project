# Dataset setup

Place the ISOT dataset files in this folder:

- `Fake.csv`
- `True.csv`

The loader expects each file to contain a `text` column and uses the `title` column when present. It assigns fake = `0` and real = `1`, removes empty and normalized duplicate articles, and reports duplicate text with conflicting labels for manual review.

Run the split command from the repository root. It writes generated CSVs to `data/processed/`, which is excluded from Git:

```powershell
python -m src.split_data --data-dir data --output-dir data/processed --seed 42
```

Keep the dataset files and generated splits out of commits. Confirm the dataset's source and usage terms with your course/project instructions.
