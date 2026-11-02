# Kaggle comparison run

This report records one Kaggle run on the shared ISOT split (5,825 test articles, seed 42). The notebook used up to five epochs for MLP, CNN, and LSTM, and two epochs with max sequence length 128 and batch size 32 for DistilBERT. Early stopping may reduce epochs. Each worker reported one visible GPU; the two-GPU notebook scheduled DistilBERT and the other models on separate GPUs.

| Model | Accuracy | Fake precision | Fake recall | Fake F1 |
|---|---:|---:|---:|---:|
| MLP | 99.31% | 99.51% | 98.99% | 99.25% |
| CNN | 98.70% | 98.30% | 98.88% | 98.59% |
| LSTM | 97.67% | 97.61% | 97.32% | 97.46% |
| DistilBERT | 99.81% | 99.85% | 99.74% | 99.80% |

Full metrics and runtime measurements are in `model_comparison.csv`. These are results from one random article split and one seed. High scores may reflect dataset-specific patterns; they do not establish real-world fake-news detection. Evaluate on source- or time-held-out data and multiple seeds before making broad claims. The CSV contains results only; model weights and dataset files are intentionally not committed.
