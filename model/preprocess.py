"""
Preprocessing pipeline for retinal fundus images.

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to
normalize lighting/contrast variation across cameras and field conditions,
plus circular crop + resize to remove black borders common in fundus photos.
"""
import cv2
import numpy as np
from PIL import Image


def crop_to_circle(img: np.ndarray) -> np.ndarray:
    """Crop out the black border around the circular fundus image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Apply CLAHE on the L channel (LAB color space) to boost lesion visibility
    without blowing out color information used elsewhere in the pipeline."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image(path: str, size: int = 224) -> np.ndarray:
    """Full preprocessing pipeline: load -> crop -> CLAHE -> resize.
    Returns an RGB uint8 numpy array ready for model input transforms."""
    img = np.array(Image.open(path).convert("RGB"))
    img = crop_to_circle(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def preprocess_array(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Same pipeline but for an already-loaded array (used at inference time
    when the image arrives via API upload rather than disk path)."""
    img = crop_to_circle(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img
