# Experiment Results and Analysis

## Run configuration

- Task: binary fake-news classification, with fake = 0 and real = 1.
- Dataset: ISOT Fake News Dataset (Fake.csv and True.csv).
- Cleaned dataset split: 27,180 training, 5,824 validation, and 5,825 test articles.
- Split: stratified 70/15/15 assignment with seed 42.
- Shared test set: all four models use the same 5,825 held-out articles.
- Models: TF-IDF MLP, text CNN, text LSTM, and fine-tuned DistilBERT.
- Metrics: accuracy, fake-class precision, fake-class recall, fake-class F1, confusion matrix, parameter count, training time per epoch, and inference time per 1,000 articles.
- Reported Kaggle budget: MLP, CNN, and LSTM up to five epochs with early stopping; DistilBERT two epochs, maximum sequence length 128, batch size 32.
- The run used a CUDA GPU. The exact hardware assignment is recorded in each run configuration where available.

## Primary comparison

| Model | Accuracy | Fake precision | Fake recall | Fake F1 | Parameters | Train seconds/epoch | Inference ms/1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 99.313% | 99.513% | 98.994% | 99.253% | 5,136,769 | 4.87 | 753.9 |
| CNN | 98.695% | 98.297% | 98.883% | 98.589% | 4,061,697 | 7.87 | 478.7 |
| LSTM | 97.665% | 97.609% | 97.318% | 97.464% | 3,980,417 | 11.87 | 477.7 |
| DistilBERT | **99.811%** | **99.851%** | **99.739%** | **99.795%** | 66,955,010 | 293.39 | 3,815.4 |

DistilBERT leads on every reported classification metric. MLP is the strongest compact alternative: its accuracy is 0.50 percentage points below DistilBERT while using about 13 times fewer parameters and about 60 times fewer training seconds per epoch. CNN and LSTM have lower inference latency in this run, but their classification scores are lower.

## Confusion-matrix interpretation

The confusion matrices use rows for true labels and columns for predicted labels, with fake first and real second.

- MLP: 2,658 fake correctly classified, 27 fake missed, 13 real incorrectly flagged, and 3,127 real correctly classified.
- CNN: 2,655 fake correctly classified, 30 fake missed, 46 real incorrectly flagged, and 3,094 real correctly classified.
- LSTM: 2,613 fake correctly classified, 72 fake missed, 64 real incorrectly flagged, and 3,076 real correctly classified.
- DistilBERT: 2,678 fake correctly classified, 7 fake missed, 4 real incorrectly flagged, and 3,136 real correctly classified.

DistilBERT makes the fewest errors in both directions. LSTM has the largest number of missed fake articles and false fake alarms among the four models.

## Training behavior

The training plot shows that the MLP and DistilBERT losses fall quickly, while the CNN and LSTM improve more gradually. The LSTM validation curve is less stable than the other models. These observations describe one run; repeated seeds would be needed to estimate run-to-run variation.

## Conclusion

For this split and seed, DistilBERT is the best-performing model, with 99.811% accuracy and 99.795% fake-class F1. MLP offers the best measured accuracy-versus-training-cost trade-off for a lower-resource deployment. The results show that the ISOT classes are highly separable, but they do not prove factual verification or generalization to new publishers, events, or time periods.

## Limitations and responsible interpretation

This experiment uses one dataset, one random split, and one seed. The dataset may contain source, writing-style, or event-specific cues that allow high scores without checking claims against evidence. No source-held-out evaluation, time-held-out evaluation, multiple-seed summary, human error categorization, calibration analysis, or external fact-checking evaluation was performed.

BLEU and IoU were not used because this is a binary text-classification task. Accuracy, precision, recall, F1, and confusion matrices are the appropriate reported metrics here.

## Figures and artifacts

- [Comparison CSV](model_comparison.csv)
- [Comparison JSON](model_comparison.json)
- [Confusion matrices](confusion_matrices.png)
- [Training and held-out F1 plot](training_and_f1.png)
- [Split metadata](split_metadata.json)
- Per-model configurations and metrics are under [artifacts](../artifacts/).
