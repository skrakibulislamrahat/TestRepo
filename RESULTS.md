# Locked Result Summary

These values are copied from the finalized project result summary and define the primary claims for this repository.

## Nominal pneumonia transport

**Kaggle Pneumonia → CheXpert Pneumonia**

- AUROC: **0.690**
- Balanced accuracy: **0.574**
- ECE-15: **0.234**
- HCER@0.90: **0.199**

This result supports weak cross-dataset transport despite a nominally similar pneumonia label.

## Opacity-aligned transport

**CheXpert Lung Opacity → CheXpert Consolidation**

- AUROC: **0.927**
- Balanced accuracy: **0.860**
- ECE-15: **0.069**
- HCER@0.90: **0.009**

**RSNA Lung Opacity → CheXpert Consolidation**

- AUROC: **0.885**
- Balanced accuracy: **0.737**
- ECE-15: **0.112**
- HCER@0.90: **0.026**

These results support stronger transport when source/target semantics are aligned around opacity/consolidation.

## High-confidence failure under transport

**Kaggle Pneumonia → RSNA Lung Opacity**

- AUROC: **0.802**
- Balanced accuracy: **0.657**
- ECE-15: **0.476**
- HCER@0.90: **0.337**

**Kaggle Pneumonia → CheXpert Consolidation**

- AUROC: **0.830**
- Balanced accuracy: **0.621**
- ECE-15: **0.407**
- HCER@0.90: **0.325**

These experiments show that reasonable discrimination can coexist with severe probability/reliability failure after transport.

## Interpretation guardrails

The project does **not** support claims that:

- pneumonia AI universally fails;
- one dataset is clinically superior to another;
- similarly named labels are necessarily equivalent;
- CheXpert Composite is a primary benchmark result.

Thresholds used for operating-point transfer were selected using source validation predictions only. Target labels were not used for threshold tuning.
