# Reproducibility Guide

## Public workflow

The original research was developed as a phased Google Colab workflow. The GitHub version deliberately converts the core modeling/evaluation logic into normal Python scripts instead of publishing the entire exploratory notebook history.

Public implementation:

```text
src/train_source.py
src/evaluate_transfer.py
src/operating_points.py
```

The full experimental workspace covered:

1. path/data audit;
2. dataset integrity audit;
3. formal label ontology;
4. clean manifest construction;
5. leakage-safe splitting;
6. five-seed DenseNet-121 training for four source tasks;
7. cross-dataset semantic-transfer evaluation;
8. source-validation operating-point transfer;
9. CheXpert semantic-alignment audit;
10. calibration/high-confidence failure analysis;
11. semantic/provenance/shortcut audit.

Manuscript assembly, writing prompts, transfer packages, cached notebook outputs, and one-off recovery cells are intentionally excluded from GitHub.

## Environment

The original experiments were executed in Google Colab with GPU acceleration. The public scripts use standard Python/PyTorch packages:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training but the evaluation utilities also support CPU execution.

## Required split/manifest format

`src/train_source.py` expects a leakage-safe CSV containing:

```text
image_id,image_path,label,split,split_group_id
```

`split` must contain `train` and `val` rows. `split_group_id` should identify the patient/study grouping used to prevent leakage when the source dataset provides such grouping information.

`src/evaluate_transfer.py` expects at least:

```text
image_id,image_path,label
```

Use `--subset test` when evaluating a split CSV with a held-out test subset.

## Locked training protocol

The research workflow used:

- DenseNet-121;
- ImageNet initialization;
- image size 224×224;
- batch size 32;
- AdamW;
- learning rate `1e-4`;
- weight decay `1e-4`;
- maximum 5 epochs;
- early-stopping patience 3;
- random seeds `42, 1337, 2025, 7, 99`.

The generalized training script uses these values as defaults.

## Evaluation discipline

For transferred operating points, thresholds are selected using **source validation predictions only**. The chosen threshold is then applied unchanged to the target predictions. Target labels are used for evaluation, not threshold optimization.

`src/operating_points.py` implements four source-side policies:

- fixed 0.5;
- maximum Youden index;
- maximum F1;
- sensitivity ≥ 0.90 with maximum specificity, when available.

Primary cross-dataset reporting emphasizes:

- AUROC;
- balanced accuracy;
- ECE with 15 bins;
- Brier score;
- high-confidence error at confidence ≥ 0.90;
- source-validation operating-point transfer.

## Random-seed reporting

Run all five seeds for each source task. Retain seedwise predictions and metrics before calculating aggregate summaries. Do not report a single favorable seed as the project result.

## Data and checkpoints

Raw medical images and large trained checkpoints should remain outside Git. Obtain datasets from their official providers and preserve dataset/version information in the local experiment record.

## Result integrity

The numeric claims in [`RESULTS.md`](RESULTS.md) are copied from the finalized project result summary. Do not replace them with numbers from intermediate experiments without documenting a new experiment version and regenerating the full evaluation matrix.
