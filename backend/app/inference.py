"""
Loads the trained model once at startup and exposes a single `predict()`
function the API route calls per request.
"""
import base64
import gc
import io
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Restrict CPU threads to prevent memory explosion on containerized cloud tiers
torch.set_num_threads(1)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "model"))
from preprocess import preprocess_array  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
from train import build_model, NUM_CLASSES, IMG_SIZE  # noqa: E402

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]

# Plain-language notes for the non-clinical user (e.g. a rural health worker)
CLASS_EXPLANATIONS = {
    0: "No signs of diabetic retinopathy detected. Routine annual screening recommended.",
    1: "Early, mild changes detected. Recommend re-screening in 6-12 months.",
    2: "Moderate changes detected. Recommend referral to an ophthalmologist within a few months.",
    3: "Severe changes detected. Recommend prompt referral to an ophthalmologist.",
    4: "Advanced disease detected. Recommend urgent ophthalmologist referral.",
}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_gradcam = None

_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str = None):
    global _model, _gradcam
    checkpoint_path = checkpoint_path or os.path.join(
        os.path.dirname(__file__), "..", "..", "model", "checkpoints", "best_model.pt"
    )
    # Build empty model without downloading redundant ImageNet weights
    _model = build_model(pretrained=False)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(state_dict)
        del state_dict
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"WARNING: no checkpoint found at {checkpoint_path} — using untrained weights (dev only)")

    # Freeze model weights to reduce autograd memory during inference
    for param in _model.parameters():
        param.requires_grad = False

    _model.to(_device)
    _model.eval()
    gc.collect()

    # EfficientNet-B0's last conv block, standard Grad-CAM target for this arch
    target_layer = _model.features[-1]
    _gradcam = GradCAM(_model, target_layer)
    return _model


def _image_to_b64(img: np.ndarray) -> str:
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict(image_bytes: bytes) -> dict:
    """Runs the full pipeline: preprocess -> predict -> Grad-CAM overlay.
    Returns a JSON-serializable dict for the API response."""
    if _model is None:
        load_model()

    bio = io.BytesIO(image_bytes)
    pil_img = Image.open(bio)
    # Use draft() to decode directly at reduced size if JPEG (saves up to 90% memory)
    if hasattr(pil_img, "draft"):
        try:
            pil_img.draft("RGB", (512, 512))
        except Exception:
            pass
    pil_img = pil_img.convert("RGB")
    del bio

    # Downsample high-res camera photos to max 512px (more than 2x model input)
    if max(pil_img.size) > 512:
        pil_img.thumbnail((512, 512), Image.Resampling.BILINEAR)

    raw = np.array(pil_img)
    del pil_img

    processed = preprocess_array(raw, size=IMG_SIZE)
    del raw

    input_tensor = _normalize(processed).unsqueeze(0).to(_device)
    heatmap, pred_class, probs = _gradcam.generate(input_tensor)
    overlay = overlay_heatmap(processed, heatmap)

    result = {
        "predicted_class": pred_class,
        "predicted_label": CLASS_NAMES[pred_class],
        "confidence": float(probs[pred_class]),
        "class_probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "explanation": CLASS_EXPLANATIONS[pred_class],
        "processed_image_b64": _image_to_b64(processed),
        "heatmap_overlay_b64": _image_to_b64(overlay),
    }

    # Clean up intermediate buffers and trigger garbage collection
    del processed, input_tensor, heatmap, overlay
    gc.collect()

    return result

