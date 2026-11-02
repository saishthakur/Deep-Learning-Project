# Contributing

## Setup

Use Python 3.10 or newer. Follow the installation and data preparation steps in the README. Do not commit the ISOT dataset, generated article-text splits, model checkpoints, virtual environments, or Kaggle secrets.

## Changes

- Keep model comparisons on the shared split and record any changed hyperparameters.
- Do not use the held-out test set for tuning.
- Run the synthetic checks before opening a pull request:

```bash
python -m unittest discover -s tests -v
```

- Include a concise description of the change and the checks performed.
- Keep experiment outputs reproducible and document their seed and data split.
