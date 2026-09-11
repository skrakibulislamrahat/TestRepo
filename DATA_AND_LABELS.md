# Data and Label Semantics

## Datasets

The project uses three public chest X-ray benchmark families:

- **Kaggle Chest X-Ray Pneumonia** — pneumonia-oriented image classification.
- **RSNA Pneumonia Detection Challenge** — lung-opacity annotations used to construct the `RSNA_LungOpacity` task.
- **CheXpert** — multi-label chest-radiograph annotations used to define Pneumonia, Lung Opacity, Consolidation, and an audit-only composite target.

Raw images are not redistributed. Obtain each dataset from its official distribution and comply with its terms.

## Why an ontology is necessary

A central premise of this project is that a shared or similar English label does not guarantee an identical prediction target. Dataset labels may differ in annotation process, operational definition, prevalence, acquisition context, and relationship to radiographic findings.

The pipeline therefore creates an explicit label ontology before training or cross-dataset evaluation. It distinguishes:

- **Pneumonia** as a nominal diagnostic/finding label;
- **Lung Opacity** as a broader radiographic finding family;
- **Consolidation** as an opacity-related radiographic finding;
- **Composite** targets as audit constructs rather than interchangeable clinical labels.

## Uncertain CheXpert labels

The experimental workspace generates explicit manifestations for different uncertainty-handling policies rather than silently mapping uncertain values. Primary comparisons should state the uncertainty policy used for the corresponding target.

## Leakage control

Train/validation/test splits are created before model fitting and are stored separately. Cross-dataset operating-point evaluation uses thresholds chosen on the **source validation set only**. Target labels are not used to optimize the decision threshold.

## Identifiers and privacy

The repository does not distribute patient images or source medical datasets. Public code should be run against locally obtained benchmark data. Do not commit private paths, credentials, protected health information, or unlicensed image files.
