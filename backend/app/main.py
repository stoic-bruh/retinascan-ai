"""
RetinaScan AI — FastAPI backend.

Run locally:
    uvicorn app.main:app --reload --port 8000

Docs auto-generated at http://localhost:8000/docs
"""
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import load_model, predict

app = FastAPI(
    title="RetinaScan AI",
    description="Explainable AI screening for Diabetic Retinopathy (SIH26038)",
    version="0.1.0",
)

# Wide-open CORS for hackathon demo purposes. Tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        del image_bytes
        gc.collect()

    return result


# Mount frontend static files so backend + frontend can be served together in a single deployment
from pathlib import Path
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
