# RetinaScan AI

**SIH26038 — Explainable AI for Diabetic Retinopathy Screening**
Team: [add names] · Theme: MedTech/HealthTech · Sponsor: MathWorks

An AI screening tool that grades diabetic retinopathy severity from retinal
fundus images and shows *why* — via Grad-CAM heatmaps — so the result is
trustworthy to a clinician or health worker, not a black box.

## Architecture

```
[Fundus image] → [CLAHE preprocessing] → [EfficientNet-B0 classifier]
                                                   ↓
                            [Grad-CAM heatmap] + [Severity grade + confidence]
                                                   ↓
                              [FastAPI backend] → [Web frontend]
```

See `docs/architecture.md` for the full diagram and `docs/decisions/` for why
each major choice (stack, dataset) was made.

## Repo structure

```
retinascan-ai/
├── docs/
│   ├── decisions/          # ADRs — one file per key architecture decision
│   ├── architecture.md
│   └── demo-script.md      # voiceover script for the demo video
├── model/
│   ├── preprocess.py       # CLAHE + crop + resize
│   ├── train.py            # EfficientNet-B0 fine-tuning
│   └── gradcam.py          # explainability heatmap generation
├── backend/
│   └── app/
│       ├── main.py         # FastAPI app
│       └── inference.py    # model loading + prediction pipeline
├── frontend/
│   └── index.html          # single-file demo UI
├── data/
│   └── README.md           # dataset download instructions (not committed)
├── ppt/                    # versioned copies of the pitch deck
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training the model

1. Download APTOS 2019 per `data/README.md`.
2. Run:
   ```bash
   cd model
   python train.py --data_dir ../data/aptos2019 --epochs 15
   ```
   Best checkpoint saves to `model/checkpoints/best_model.pt`.

## Running the app locally

```bash
# Terminal 1 — backend
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
python -m http.server 5500
# open http://localhost:5500
```

## Live deployment

[Add live URL here once deployed — Render/HuggingFace Spaces, free tier]

## Submission checklist (per community admin instructions, deadline Sept 3)

- [ ] 6-page PPT (`ppt/`)
- [ ] Demo video with voiceover
- [ ] Working code (this repo)
- [ ] Live deployment link (bonus, software track)
- [ ] README complete (this file)
