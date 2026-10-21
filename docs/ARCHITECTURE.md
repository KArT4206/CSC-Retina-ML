# Architecture

![Pipeline](figures/architecture.png)

## Preprocessing

![Preprocessing](figures/preprocessing.png)

Every image from every class and dataset takes the same path and is cached as a 224 × 224 PNG without metadata, so any difference that survives must be in the image content.

## Patient-grouped evaluation

![Grouped cross-validation](figures/grouped_cv.png)

Splits are made over patients (or near-duplicate clusters), never single images. An outer fold tests each group once, a validation subset handles early stopping and backbone ranking, and inner folds generate out-of-fold features for the meta stage. The procedure is repeated over five seeds.

## Fusion stages

| Stage | What it learns | Fitted on |
|---|---|---|
| R1 | Five fine-tuned backbones; Top-4 by validation score | Training groups |
| R2 | PCA per model (32 components) and GA weights for M2 to M4, logistic regression | Out-of-fold features |
| R3 | M1 features plus weighted R2 features, MLP head | Out-of-fold features |
| UAGF | Per-image softmax gates from each model's binary entropy | Out-of-fold probabilities |
| NSGA-II | Model weights and threshold trading sensitivity, specificity and model count | Out-of-fold probabilities |
| MC dropout | 20 stochastic passes through each Top-4 head; mean and variance | Test time |

![UAGF](figures/uagf.png)

## Web app

| Component | Role |
|---|---|
| `csc_app/app.py` | Flask routes: analyse (single or batch upload), live progress over server-sent events, report page and PDF, model and validation page |
| `csc_app/engine.py` | Loads the model bundle and runs every pipeline stage on one scan |
| `csc_app/pdf_report.py` | Builds the downloadable PDF report |
| `csc_v3/` | The research pipeline the app reuses for preprocessing, backbones and fusion |

With `CSC_PUBLIC=1`, the history page lists nothing and uploaded images are deleted when their analysis finishes, so visitors never see each other's scans.
