# Deep Learning Comparison for Fake News Detection

This project compares four model families on the ISOT Fake News Dataset: a TF-IDF multilayer perceptron (MLP), a text convolutional network (CNN), a text long short-term memory network (LSTM), and fine-tuned DistilBERT. All models use one cleaned dataset and the same stratified 70/15/15 train, validation, and test assignment. The test set is reserved for final evaluation. Predictions describe learned dataset patterns; they do not verify an article's claims.

## Requirements

Use Python 3.10 or newer. Install PyTorch for the CPU or CUDA environment, then install the remaining dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the ISOT files `Fake.csv` and `True.csv` in `data/`. The dataset is not committed to this repository; obtain it from the ISOT Fake News Dataset source and keep both CSV files in `data/`. Dataset files, generated splits, model checkpoints, and experiment outputs are excluded from Git. Start a fresh run by regenerating the split and retraining the models; previous committed MLP metrics have been removed.

## Prepare the shared data split

```powershell
python -m src.split_data --data-dir data --output-dir data/processed --seed 42
```

The pipeline removes blank and normalized duplicate articles, rejects duplicate text with conflicting labels, lowercases article text while preserving the structural `[SEP]` title/body delimiter, removes URLs and HTML, and strips common Reuters opening tags. It drops metadata fields such as subject and date, then assigns fake = 0 and real = 1. Article IDs are hashes of cleaned text. It writes train/validation/test CSVs plus a text-free manifest for reproducing the same assignment on another machine. Learned vocabularies and TF-IDF weights are fitted on the training set only.

Review cleaned examples for any remaining source or formatting shortcuts before interpreting scores. To reuse an agreed manifest:

```powershell
python -m src.split_data --data-dir data --output-dir data/processed --manifest path\to\split_manifest.csv
```

## Train models

Run each command from the repository root. The neural models use Adam, learning rate 0.001, batch size 64, at most 20 epochs, dropout, and early stopping on validation loss (patience 3). DistilBERT uses AdamW, learning rate 2e-5, batch size 16, at most 3 epochs, weight decay 0.01, linear warm-up/decay, and the same early stopping rule.

```powershell
python -m src.train_model --model mlp
python -m src.train_model --model cnn
python -m src.train_model --model lstm
python -m src.train_model --model distilbert
```

Defaults follow the report protocol: MLP uses training-only TF-IDF unigrams and bigrams (up to 20,000 terms) with 256- and 64-unit hidden layers; CNN and LSTM use a training-only vocabulary capped at 30,000 tokens, 400-token sequences, and 128-dimensional embeddings. CNN uses parallel 3-, 4-, and 5-token convolutions with 128 filters each. LSTM is unidirectional with 128 hidden units. DistilBERT defaults to `distilbert/distilbert-base-uncased`, maximum length 256, and three epochs. All commands use seed 42 by default. Options such as `--batch-size`, `--learning-rate`, `--epochs`, `--patience`, `--max-length`, `--model-name`, and `--seed` allow controlled experiments; record any changes when comparing results.

Each run writes its best validation-loss checkpoint, configuration, split IDs/fingerprints, parameter count, and epoch history under `artifacts/<model>/`. The first DistilBERT run downloads model files and needs network access.

## Evaluate and compare

Evaluate each saved model once on the held-out test split:

```powershell
python -m src.evaluate_model --model mlp
python -m src.evaluate_model --model cnn
python -m src.evaluate_model --model lstm
python -m src.evaluate_model --model distilbert
python -m src.compare_models --artifact-root artifacts --output-dir results
```

Each evaluation reports accuracy, precision, recall, and F1 with fake (label 0) as the positive class, the confusion matrix, parameter count, mean training seconds per epoch, inference milliseconds per 1,000 articles, and the IDs and text excerpts of misclassified examples. The comparison command creates `results/model_comparison.csv`, `results/model_comparison.json`, `results/training_and_f1.png`, and `results/confusion_matrices.png`. Use the error examples to make a human-reviewed error analysis; the code does not assign speculative error categories automatically.

For a fair comparison, run all models on the same split and hardware, and do not select settings using test results. A single seed and a random article split do not establish robust performance or generalization to new publishers, events, or time periods. Consider a source- or time-held-out evaluation and multiple seeds before drawing broad conclusions.

## Tests

The synthetic data and split checks do not require the actual dataset or a downloaded Transformer:

```powershell
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs these checks on Python 3.10 and 3.12. The checks use synthetic CSVs and do not download the real dataset. Both versions passed on the latest data-normalization fix.

