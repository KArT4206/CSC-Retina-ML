# Results

## Backbones (R1, same-scanner, mean ± SD over 5 seeds × 4 folds)

| Backbone | AUC | Balanced accuracy | Times in Top-4 (of 20) |
|---|---|---|---|
| ResNet-34 | 0.940 ± 0.046 | 0.870 ± 0.082 | 20 |
| ResNet-50 | 0.887 ± 0.075 | 0.783 ± 0.089 | 16 |
| DenseNet-121 | 0.950 ± 0.038 | 0.889 ± 0.063 | 19 |
| EfficientNet-B0 | 0.898 ± 0.115 | 0.801 ± 0.135 | 17 |
| MobileNetV3-Large | 0.846 ± 0.086 | 0.680 ± 0.122 | 8 |

![Learning curves](figures/learning_curves.png)

## Fusion and ensembles

| Method | AUC per seed | Balanced accuracy | Sensitivity | Specificity | Matched-type AUC |
|---|---|---|---|---|---|
| R1 M1 alone | 0.932 ± 0.018 | 0.849 ± 0.025 | 0.894 | 0.805 | 0.792 |
| R2 GA-weighted M2–M4 | 0.851 ± 0.052 | 0.768 ± 0.045 | 0.751 | 0.786 | 0.723 |
| R3 full fusion | 0.700 ± 0.110 | 0.646 ± 0.082 | 0.735 | 0.557 | 0.735 |
| Average ensemble | 0.954 ± 0.010 | 0.909 ± 0.016 | 0.885 | 0.933 | 0.783 |
| Temperature-scaled ensemble | 0.938 ± 0.016 | 0.890 ± 0.032 | 0.865 | 0.914 | 0.756 |
| GA probability weights | 0.903 ± 0.044 | 0.834 ± 0.068 | 0.820 | 0.848 | 0.688 |
| UAGF | 0.913 ± 0.037 | 0.854 ± 0.047 | 0.823 | 0.886 | 0.729 |
| NSGA-II balanced point | 0.924 ± 0.018 | 0.878 ± 0.027 | 0.855 | 0.900 | 0.774 |
| NSGA-II screening point | 0.924 ± 0.022 | 0.849 ± 0.021 | 0.898 | 0.800 | 0.735 |
| MC-dropout ensemble | 0.954 ± 0.009 | 0.913 ± 0.015 | 0.887 | 0.938 | 0.783 |
| Cross-attention fusion | 0.734 ± 0.102 | 0.655 ± 0.084 | 0.576 | 0.733 | 0.723 |

![ROC and reliability](figures/roc_reliability.png)

![Confusion matrices](figures/confusion.png)

## NSGA-II operating points

![Pareto front](figures/nsga2_pareto.png)

The screening point reached sensitivity 0.898 and the balanced point specificity 0.900, against 0.894 and 0.805 for the best single model.

## Refer to clinician

| Uncertainty | 100% kept | 90% | 80% | 70% |
|---|---|---|---|---|
| MC-dropout variance | 0.913 | 0.926 | 0.936 | 0.949 |
| UAGF entropy | 0.854 | 0.882 | 0.901 | 0.914 |

![Referral curve](figures/refer_to_clinician.png)

## Shortcut checks

| Check | Result |
|---|---|
| OCT-C8: random forest on 16 × 16 thumbnails | AUC 0.998 |
| OCT-C8: raw image size alone | AUC 1.000 |
| OCT-C8: CNN trained on thumbnails only | AUC 1.000 |
| Same-scanner: CNN trained on thumbnails only | AUC 0.867 |
| Same-scanner: shuffled training labels (M1, R3) | AUC 0.437, 0.502 |
| Same-scanner: matched-type AUC (screenshots only) | about 0.78 |

## Grad-CAM

![Grad-CAM, same-scanner](figures/gradcam_same_scanner.png)

On the same-scanner data, 0.37 of the Grad-CAM mass falls on retinal tissue that covers 0.20 of the image, about 1.8 times its area share.

![Grad-CAM, OCT-C8](figures/gradcam_octc8.png)

On OCT-C8 the maxima sit on an export overlay marker, which supports the shortcut finding.

## Cross-dataset

| Trained on → tested on | AUC |
|---|---|
| Same-scanner → same-scanner (in-domain) | 0.912 |
| Same-scanner → OCTID | 0.435 |
| Same-scanner → OCT-C8 | 0.600 |
| OCT-C8 → same-scanner screenshots | 0.711 |
