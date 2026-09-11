# Same Label, Different Disease

## Semantic transport failure in pneumonia and lung-opacity classification across chest X-ray benchmarks

This repository contains a reliability-focused medical-imaging study asking a simple but important question:

> **When two chest X-ray datasets use labels that sound clinically similar, do models trained on those labels actually learn transferable versions of the same task?**

The experiments compare models trained on **Kaggle Chest X-Ray Pneumonia**, **RSNA Lung Opacity**, and **CheXpert** targets. The central result is that label names alone are not enough to establish semantic equivalence: transfer behavior depends strongly on the label family and dataset provenance.

## Primary findings

| Source → target | AUROC | Balanced accuracy | ECE-15 | HCER@0.90 |
|---|---:|---:|---:|---:|
| Kaggle Pneumonia → CheXpert Pneumonia | 0.690 | 0.574 | 0.234 | 0.199 |
| CheXpert Lung Opacity → CheXpert Consolidation | 0.927 | 0.860 | 0.069 | 0.009 |
| Kaggle Pneumonia → RSNA Lung Opacity | 0.802 | 0.657 | 0.476 | 0.337 |
| Kaggle Pneumonia → CheXpert Consolidation | 0.830 | 0.621 | 0.407 | 0.325 |
| RSNA Lung Opacity → CheXpert Consolidation | 0.885 | 0.737 | 0.112 | 0.026 |

The strongest interpretation supported by the experiments is **not** that pneumonia AI universally fails. It is that superficially similar dataset labels should not be treated as interchangeable clinical prediction targets without explicit semantic and transport evaluation.

## Experimental design

The public workflow includes:

1. dataset availability and integrity auditing;
2. a formal label ontology across source datasets;
3. clean per-dataset manifests;
4. leakage-safe train/validation/test splits;
5. DenseNet-121 training across **five random seeds** for four source tasks;
6. cross-dataset semantic-transfer evaluation;
7. source-validation operating-point transfer with **no target-label threshold tuning**;
8. CheXpert semantic-alignment analysis;
9. calibration and high-confidence error analysis;
10. semantic/provenance/shortcut auditing.

### Source tasks

- `Kaggle_Pneumonia`
- `RSNA_LungOpacity`
- `CheXpert_Pneumonia`
- `CheXpert_LungOpacity`

### Important audit targets

- `CheXpert_Consolidation`
- `CheXpert_Composite` — audit-only because of near-degenerate balance; it is not used as a primary result.

## Metrics

The study emphasizes metrics that directly address transport reliability:

- AUROC
- balanced accuracy
- Expected Calibration Error (15 bins)
- high-confidence error rate at 0.90 confidence (`HCER@0.90`)
- operating-point transfer from source validation to target data

AUPRC is not treated as a primary cross-dataset claim on highly imbalanced CheXpert targets.

## Repository structure

```text
.
├── notebooks/
│   └── Same_Label_Different_Disease.ipynb
├── RESULTS.md
├── REPRODUCIBILITY.md
├── DATA_AND_LABELS.md
├── CITATION.cff
├── requirements.txt
└── README.md
```

The notebook has cached outputs, execution counts, and manuscript-writing/transfer-package cells removed. The public version ends at the scientific audit stages.

## Datasets

Raw chest X-ray datasets are **not** redistributed in this repository. Users must obtain each dataset from its official source and comply with the corresponding terms of use. See [`DATA_AND_LABELS.md`](DATA_AND_LABELS.md).

## Research status

This repository documents an active research project. A formal article citation will be added only when stable publication metadata is available.

## Responsible use

This code is for research and reproducibility. It is not a clinical diagnostic system and should not be used for patient-care decisions without appropriate validation, governance, and regulatory review.
