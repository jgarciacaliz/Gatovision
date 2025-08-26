import hashlib
import os
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from .config import MEMORY_IMAGES


def ensure_dirs():
    MEMORY_IMAGES.mkdir(parents=True, exist_ok=True)


def save_crop(identity: str, img: np.ndarray) -> Path:
    """Saves a cropped image of the identified subject."""
    ensure_dirs()
    ident_dir = MEMORY_IMAGES / identity
    ident_dir.mkdir(parents=True, exist_ok=True)
    filename = f"crop_{hashlib.md5(os.urandom(8)).hexdigest()[:8]}.jpg"
    out = ident_dir / filename
    cv2.imwrite(str(out), img)
    return out


def crop_from_bbox(img: np.ndarray, xyxy: Tuple[int, int, int, int], pad: float = 0.05) -> np.ndarray:
    """Crops the image given a bounding box with optional padding."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    xp = int(bw * pad)
    yp = int(bh * pad)
    x1 = max(0, x1 - xp)
    y1 = max(0, y1 - yp)
    x2 = min(w, x2 + xp)
    y2 = min(h, y2 + yp)
    crop = img[y1:y2, x1:x2]
    return crop


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Converts an image from BGR to RGB color space."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def normalize_imagenet(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Normalizes the image to be compatible with the ImageNet dataset."""
    h, w = img_rgb.shape[:2]
    scale = size / min(h, w)
    img = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
    h2, w2 = img.shape[:2]
    y0 = (h2 - size) // 2
    x0 = (w2 - size) // 2
    img = img[y0:y0 + size, x0:x0 + size]
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (0, 1, 2))
    img = np.transpose(img, (2, 0, 1))
    return img
