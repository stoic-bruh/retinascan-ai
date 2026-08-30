# ADR 0001: Model & Backend Stack — Python (not MATLAB)

**Status:** Accepted
**Date:** 2026-08-28

## Context
PS SIH26038 (Explainable AI for Diabetic Retinopathy Screening) is sponsored by
MathWorks. MathWorks-sponsored PS typically favor MATLAB/Simulink, and MATLAB's
Deep Learning Toolbox has a built-in `gradCAM` function that maps directly to
our explainability requirement.

We considered three options:
1. **MATLAB only** — model + Grad-CAM + app in MATLAB.
2. **Hybrid** — MATLAB for model training/Grad-CAM, Python for the deployed app
   (via ONNX export).
3. **Python only** — PyTorch/TensorFlow + `pytorch-grad-cam`, FastAPI backend,
   web frontend, deployed live.

## Decision
We chose **Option 3: Python only.**

## Reasoning
- No team member has MATLAB or a Deep Learning Toolbox license, and no campus-wide
  license was available in the time we had to check.
- The submission rubric explicitly rewards **live deployment** for the software
  track (bonus points) — a Python/FastAPI stack deploys to free hosting
  (Render/HF Spaces) in minutes; a MATLAB-based app does not.
- `pytorch-grad-cam` provides functionally equivalent Grad-CAM output to
  MATLAB's `gradCAM`, so the explainability deliverable is unaffected.
- Conceptually we are still following the CNN + Grad-CAM workflow MathWorks'
  own materials document — only the implementation language changed.

## Consequences
- PPT slide 4 (Technical Approach) updated to reflect Python/PyTorch instead
  of MATLAB, while keeping the same architecture diagram shape (Image →
  Preprocess → CNN → Grad-CAM → Report).
- If a team member obtains a MATLAB license later, model training can be
  redone in MATLAB and exported to ONNX for the same Python-based backend —
  this ADR would then be superseded, not rewritten.
