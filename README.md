# Comparative Analysis of Deep Learning Architectures for Fake News Detection

This project compares deep-learning models that classify English news articles as **fake (0)** or **real (1)** using the ISOT Fake News Dataset. Planned models are MLP, CNN, LSTM, and a Transformer. The initial Transformer implementation fine-tunes DistilBERT.

> **Status:** Shared dataset loading/splitting and the DistilBERT train, evaluate, and predict scripts are now in `src/`. MLP, CNN, and LSTM implementations and cross-model result collection remain team work.

A classifier learns patterns associated with labels in its training data. Its output is a prediction, not independent verification that an article's claims are true.

## Dataset and leakage prevention

Place `Fake.csv` and `True.csv` under `data/`; see [data/README.md](data/README.md). Do not commit dataset files or generated splits.

The shared loader in `src/data.py` expects a `text` column and uses `title` when present. It assigns fake = `0`, real = `1`, removes empty rows and normalized duplicate text, and stops if identical normalized text appears with conflicting labels so it can be reviewed. The split script makes stratified 70%/15%/15% train/validation/test CSVs with a fixed seed.

Use these same split files and labels for all four models. Inspect possible source/formatting artifacts, keep duplicates together, fit learned preprocessing only on training data, choose settings using validation data, and reserve test data for final evaluation. Agree with teammates before changing the shared split procedure.

## Setup

Use Python 3.10 or newer. Install PyTorch for the CPU or CUDA environment you intend to use, then install the remaining project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first model run downloads the pretrained tokenizer and weights from Hugging Face and therefore needs network access.

## Prepare the shared split

Run from the repository root:

```powershell
python -m src.split_data --data-dir data --output-dir data/processed --seed 42
```

This writes `train.csv`, `validation.csv`, and `test.csv` to `data/processed/`. Keep these generated files local or share the exact split through your team's agreed data workflow; do not commit article text.

## Train DistilBERT

```powershell
python -m src.train_transformer --train data/processed/train.csv --validation data/processed/validation.csv --model-name distilbert/distilbert-base-uncased --output-dir artifacts/distilbert --epochs 3 --batch-size 8 --learning-rate 2e-5 --max-length 256 --seed 42
```

The script selects the checkpoint with the best validation F1 for the real-news class (label 1) and saves the model, tokenizer, and run settings under `artifacts/distilbert/`. Change batch size or maximum length to fit available memory, and record those changes when comparing models. Long articles are truncated at the selected maximum length.

## Evaluate and predict

Evaluate the saved model on the held-out test split:

```powershell
python -m src.evaluate --model-dir artifacts/distilbert --split data/processed/test.csv
```

The script prints and saves accuracy, precision, recall, F1 for real news, a confusion matrix (label order fake then real), and a classification report to `artifacts/distilbert/test_metrics.json`.

Classify a sample article:

```powershell
python -m src.predict --model-dir artifacts/distilbert --title "Example headline" --text "Article text goes here"
```

The output includes predicted label and class probabilities. Treat these as model scores, not proof that a story is true or false.

## Comparing models

For each architecture, use the same train/validation/test assignments and report accuracy, precision, recall, F1, confusion matrix, input fields, seed, and model settings. State the F1 averaging/positive-label convention. Use validation results to select settings; do not tune against test results. Collect final results in a shared table such as `results/model_comparison.csv` and note any differences that affect fairness.

The MLP, CNN, and LSTM modules and the consolidated comparison table have not been added yet. Coordinate their input and evaluation interfaces with the team.

## Project layout

```text
data/                    Local ISOT CSVs and generated splits (ignored by Git)
src/data.py              Shared data loading and duplicate checks
src/split_data.py        Deterministic stratified split
src/train_transformer.py DistilBERT fine-tuning
src/evaluate.py          Held-out evaluation
src/predict.py           Single article prediction
requirements.txt         Python dependencies
artifacts/               Local checkpoints and metrics (ignored by Git)
```

## Git workflow

Keep work on a feature branch, review the diff, and open a pull request for team review:

```powershell
git switch -c feature/your-change
git status
git add README.md data/README.md src requirements.txt .gitignore
git commit -m "Describe your change"
git push -u origin feature/your-change
```

Check `git status` before committing. Dataset CSVs, generated splits, model weights, virtual environments, and secrets should remain out of Git.
