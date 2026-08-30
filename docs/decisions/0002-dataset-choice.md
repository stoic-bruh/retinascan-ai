# ADR 0002: Dataset Selection

**Status:** Accepted
**Date:** 2026-08-28

## Context
We need labeled retinal fundus images with DR severity grades to train a
classifier, plus a way to validate that Grad-CAM heatmaps highlight genuine
lesions (not spurious image artifacts).

## Options considered
- **APTOS 2019 Blindness Detection** (Kaggle) — ~3,662 labeled fundus images,
  5-class severity (0=No DR ... 4=Proliferative DR).
- **IDRiD** (Indian Diabetic Retinopathy Image Dataset) — smaller, but includes
  pixel-level lesion annotations (microaneurysms, hemorrhages, exudates).
- **Messidor-2** — ~1,748 images, DR grades, widely used as an external
  validation set in DR literature.
- **DRIVE** — vessel segmentation dataset, not severity-labeled; not directly
  useful for our classification task.

## Decision
- **Primary training set:** APTOS 2019 — largest, cleanest labels, most
  reference implementations to sanity-check against.
- **Grad-CAM validation set:** IDRiD's lesion-annotated subset — used only to
  confirm heatmaps activate on actual lesions, not to train the classifier.
- **Held-out external check (stretch goal):** Messidor-2, if time allows, to
  show the model isn't overfit to APTOS's specific imaging conditions.
- **DRIVE:** not used — out of scope for severity classification.

## Consequences
- Data folder (`data/`) stores only download instructions and licenses, not
  raw images (Kaggle ToS + repo size).
- Class imbalance in APTOS (heavily skewed toward "No DR") must be corrected
  via weighted loss and/or oversampling — tracked in `model/train.py`.
