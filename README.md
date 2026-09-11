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

The full research workspace in Drive includes:

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

The GitHub repository contains a cleaned, reusable implementation of the core modeling and transport-evaluation stages rather than dumping the full exploratory Colab history.

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

## Clean public implementation

```text
src/
├── train_source.py        # DenseNet-121, five seeds, source train/validation
├── evaluate_transfer.py   # target evaluation, AUROC, calibration, HCER
└── operating_points.py    # select thresholds on source validation only and transfer unchanged
```

### Example: train one source task

```bash
python src/train_source.py \
  --split-csv /path/to/kaggle_pneumonia_splits.csv \
  --source-task Kaggle_Pneumonia \
  --output-dir ./runs
```

Defaults reproduce the research protocol: DenseNet-121 with ImageNet initialization; seeds `42 1337 2025 7 99`; 224×224 images; batch size 32; AdamW; learning rate `1e-4`; weight decay `1e-4`; five maximum epochs; early-stopping patience 3.

### Example: transport to a target dataset

```bash
python src/evaluate_transfer.py \
  --checkpoint ./runs/Kaggle_Pneumonia/seed_42/best_checkpoint.pt \
  --target-csv /path/to/rsna_lungopacity_splits.csv \
  --subset test \
  --source-task Kaggle_Pneumonia \
  --target-task RSNA_LungOpacity \
  --output-prefix ./transfer/kaggle_to_rsna_seed42
```

### Example: transfer a source-selected operating point

```bash
python src/operating_points.py \
  --source-validation ./runs/Kaggle_Pneumonia/seed_42/val_predictions.csv \
  --target-predictions ./transfer/kaggle_to_rsna_seed42_predictions.csv \
  --output ./transfer/kaggle_to_rsna_seed42_operating_points.csv
```

The threshold policies are selected using **source validation labels only** and then applied unchanged to the target predictions.

## Repository structure

```text
.
├── src/
│   ├── train_source.py
│   ├── evaluate_transfer.py
│   └── operating_points.py
├── RESULTS.md
├── REPRODUCIBILITY.md
├── DATA_AND_LABELS.md
├── CITATION.cff
├── requirements.txt
└── README.md
```

The unpublished manuscript-writing package, cached notebook outputs, model checkpoints, and raw medical-image datasets are intentionally not published here.

## Data

Raw chest X-ray datasets are **not** redistributed. Users must obtain each dataset from its official source and comply with its terms. See [`DATA_AND_LABELS.md`](DATA_AND_LABELS.md).

## Research status

This repository documents an active research project. A formal article citation will be added only when stable publication metadata is available.

## Responsible use

This code is for research and reproducibility. It is not a clinical diagnostic system and should not be used for patient-care decisions without appropriate validation, governance, and regulatory review.
