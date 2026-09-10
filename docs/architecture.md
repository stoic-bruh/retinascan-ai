# Architecture

## Data flow

```mermaid
flowchart LR
    A[Fundus image upload] --> B[CLAHE preprocessing<br/>crop + contrast normalize]
    B --> C[EfficientNet-B0<br/>fine-tuned on APTOS 2019]
    C --> D[Severity grade 0-4<br/>+ confidence scores]
    C --> E[Grad-CAM<br/>heatmap generation]
    D --> F[FastAPI response]
    E --> F
    F --> G[Web frontend<br/>image + heatmap + explanation]
```

## Model
- Backbone: EfficientNet-B0, ImageNet-pretrained, fine-tuned end-to-end.
- Loss: weighted cross-entropy (class weights from `sklearn.compute_class_weight`)
  to correct APTOS's imbalance toward "No DR".
- Input: 224x224 RGB, CLAHE-normalized.

## Explainability
- Grad-CAM on the last convolutional block of EfficientNet-B0.
- Validated qualitatively against IDRiD lesion masks — heatmap activation
  should overlap with annotated microaneurysms/hemorrhages/exudates, not
  random background.

## Serving
- FastAPI exposes a single `/predict` endpoint (multipart image upload →
  JSON response with grade, confidence, per-class probabilities, base64
  processed image, base64 heatmap overlay).
- Frontend is a static single-file HTML/JS page — no build step, easy to
  demo from any machine.

## Deployment
- Backend: Render (free tier) — see README for live URL.
- Model checkpoint loaded once at startup, kept in memory for the process
  lifetime (no per-request disk I/O).
