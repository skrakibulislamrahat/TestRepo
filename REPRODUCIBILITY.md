# Reproducibility Guide

## Public workflow

The notebook `notebooks/Same_Label_Different_Disease.ipynb` contains the research pipeline through Phase 9:

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

Manuscript-assembly and manuscript-transfer cells from the working notebook are intentionally excluded from the public version.

## Environment

The original experiments were executed in Google Colab with GPU acceleration. The public notebook uses standard Python/PyTorch packages plus Colab Drive mounting. Install the core dependencies with:

```bash
pip install -r requirements.txt
```

## Project root

The notebook defaults to the original Colab location:

```text
/content/drive/MyDrive/KJR_Pneumonia_Semantic_Transport_Project
```

For a different location, set the environment variable before running cells:

```python
import os
os.environ["PROJECT_ROOT"] = "/your/project/root"
```

The public notebook resolves `PROJECT_ROOT` through this variable where the working notebook originally used the fixed Colab path.

## Datasets

The raw datasets are not part of this repository. Obtain Kaggle Chest X-Ray Pneumonia, RSNA Pneumonia Detection Challenge data, and CheXpert from their official sources. The Phase 0/1 cells audit the supplied locations before later stages execute.

## Random seeds

The training workflow evaluates multiple independent seeds:

```text
42, 1337, 2025, 7, 99
```

Seedwise results are retained before aggregation so cross-dataset conclusions are not based on a single training run.

## Evaluation discipline

For transferred operating points, thresholds are selected using source validation predictions only. Target labels are used for evaluation, not threshold selection. This prevents target-set adaptation from being hidden inside the reported transfer results.

Primary reporting uses:

- AUROC;
- balanced accuracy;
- ECE with 15 bins;
- high-confidence error rate at confidence ≥ 0.90;
- source-validation operating-point transfer.

## Expected output organization

The working pipeline creates project subdirectories for configuration, dataset audits, label ontology, manifests, splits, models/checkpoints, predictions, metrics, figures/tables, and logs. Large outputs/checkpoints should remain outside Git unless there is a clear reason to version them.

## Public-notebook cleaning

To make the research artifact readable and safe to publish, the GitHub copy has:

- cached notebook outputs removed;
- execution counts removed;
- volatile Colab metadata removed;
- duplicate experimental cell removed;
- one-off extraction-repair cell omitted;
- manuscript-writing and transfer-package stages omitted.

The scientific analysis code itself is retained through the final semantic/provenance audit.
