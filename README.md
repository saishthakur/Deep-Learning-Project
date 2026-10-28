# Comparative Analysis of Deep Learning Architectures for Fake News Detection

This project compares deep-learning models that classify English news articles as **fake (0)** or **real (1)** using the ISOT Fake News Dataset. The planned comparison includes an MLP, CNN, LSTM, and a Transformer (BERT or DistilBERT).

> **Project status:** The repository currently contains this project plan only. Training, evaluation, and prediction scripts still need to be implemented.

## Project goals

- Prepare a consistent, reproducible dataset split for all model comparisons.
- Train and evaluate the four architectures using the same labels and held-out test examples.
- Compare accuracy, precision, recall, F1-score, and confusion matrices.
- Record model settings and results so the comparison can be reproduced.

A classifier learns patterns associated with the labels in its training data. Its output is a prediction, not independent verification that an article's claims are true.

## Dataset

Use the ISOT Fake News Dataset files `Fake.csv` and `True.csv`. Keep the CSVs on your machine; do not commit dataset files to this repository. Create `data/README.md` with the dataset source, setup instructions, and any applicable usage terms.

Expected preparation:

1. Load both CSVs and assign fake = `0`, real = `1`.
2. Check required columns and remove rows with missing article text.
3. Combine title and article text consistently for every model.
4. Remove exact duplicates and investigate near-duplicates before splitting.
5. Agree on one stratified train/validation/test split with the team (for example, 70% / 15% / 15%), and save its row IDs or assignments for every model to reuse.

Use a fixed random seed. Keep duplicate or near-duplicate articles in the same split where possible. Inspect whether source names, formatting, or other dataset artifacts reveal labels. Fit any learned preprocessing only on the training data. Do not use test results to choose model settings.

## Planned repository structure

As implementation proceeds, the repository is expected to grow toward this layout:

```text
data/                  Dataset instructions and saved split assignments
src/data.py            Dataset loading and shared preprocessing
src/split_data.py      Reproducible train/validation/test split
src/train_transformer.py
src/evaluate.py        Shared evaluation metrics
src/predict.py         Load a saved model and classify input text
configs/distilbert.yaml
tests/                 Data and inference checks
results/               Comparison results and run metadata
requirements.txt
.gitignore
```

The MLP, CNN, and LSTM implementations can live in their own modules or team-owned folders. Coordinate the shared data and evaluation interfaces before integrating them.

## Transformer work plan

Start with DistilBERT (for example, `distilbert-base-uncased`) unless the team agrees to use BERT. Fine-tune the pretrained model for binary classification. Record the model/tokenizer name, seed, maximum sequence length, batch size, learning rate, epochs, and selected checkpoint. Use validation data for model selection and evaluate the chosen model once on the held-out test set.

The tokenizer handles text segmentation and numerical encoding. Keep text cleaning minimal and use the same title/article fields as the other models. Document truncation behavior because long articles may exceed the model's maximum sequence length.

## Planned commands

The scripts below are **not implemented yet**. Update these commands when the corresponding files are added.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.split_data
python -m src.train_transformer --config configs/distilbert.yaml
python -m src.evaluate --model-dir artifacts/distilbert --split test
python -m src.predict
```

Save model checkpoints and tokenizer files under a local artifacts directory, and exclude them from Git unless the team deliberately chooses a suitable model-hosting approach.

## Evaluation and comparison

Report the following for each model on the **same test split**:

- Accuracy
- Precision, recall, and F1-score (state the averaging method)
- Confusion matrix
- Training settings and, where useful, training time

Use validation results to select hyperparameters. Put the final results in a shared table, such as `results/model_comparison.csv`, and explain differences in preprocessing or model inputs that could affect fairness.

## Git workflow

Work on a feature branch and review the changes before opening a pull request:

```powershell
git switch -c feature/distilbert-classifier
git status
git add README.md
git commit -m "Document fake news detection project plan"
git push -u origin feature/distilbert-classifier
```

Before committing, check `git status` and confirm that local datasets, model checkpoints, virtual environments, and secrets are excluded. Coordinate shared split and evaluation decisions with the team before comparing results.
