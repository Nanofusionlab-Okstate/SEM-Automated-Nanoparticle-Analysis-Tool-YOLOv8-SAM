# -*- coding: utf-8 -*-
"""
Multi-format SEM image loading and normalization utilities.

Author: Karishma Begum
Year: 2026
"""

import os
import cv2
import numpy as np


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Robustly normalize any numeric image to uint8 using percentile clipping."""
    if img is None:
        raise ValueError("Image is None")

    # If structured RGB (dtype with field names)
    if img.dtype.names is not None:
        img = np.stack([img[name] for name in img.dtype.names], axis=-1)

    # If boolean
    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255

    # If already uint8
    if img.dtype == np.uint8:
        return img

    img_float = img.astype(np.float32)

    # If multi-channel, compute percentiles on intensity (prevents weird per-channel scaling)
    if img_float.ndim == 3:
        # Use mean intensity for percentile calc
        intensity = img_float.mean(axis=2)
        p1, p99 = np.percentile(intensity, (1, 99))
    else:
        p1, p99 = np.percentile(img_float, (1, 99))

    if abs(p99 - p1) < 1e-8:
        return np.zeros_like(img_float, dtype=np.uint8)

    img_clipped = np.clip(img_float, p1, p99)
    img_norm = (img_clipped - p1) / (p99 - p1) * 255.0
    return np.clip(img_norm, 0, 255).astype(np.uint8)


def _ensure_rgb(img_u8: np.ndarray, source_hint: str = "") -> np.ndarray:
    """
    Ensure output is (H, W, 3) RGB uint8.
    source_hint: 'cv2' when loaded by OpenCV (likely BGR/BGRA),
                 'rsciio' when loaded by rosettasciio (often already RGB).
    """
    if img_u8.ndim == 2:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    if img_u8.ndim != 3:
        raise ValueError(f"Unexpected image ndim={img_u8.ndim}")

    c = img_u8.shape[2]

    # Handle alpha channel
    if c == 4:
        if source_hint == "cv2":
            # cv2 gives BGRA
            return cv2.cvtColor(img_u8, cv2.COLOR_BGRA2RGB)
        else:
            # Many non-cv2 pipelines already give RGBA
            return img_u8[:, :, :3].copy()

    if c >= 3:
        img3 = img_u8[:, :, :3]
        if source_hint == "cv2":
            # cv2 gives BGR
            return cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        else:
            # Assume already RGB
            return img3.copy()

    raise ValueError(f"Unexpected channel count: {c}")


def load(path):
    """
    Loads SEM images (TIF, DM3/DM4) and extracts pixel size.

    Returns:
        img_x (np.ndarray): 8-bit RGB image for display/AI (H,W,3)
        img   (object):      raw image / metadata placeholder (compatibility)
        pixel_size (float):  calibration scale (best-effort)
    """
    _, extension = os.path.splitext(path)
    ext = extension.lower()

    img = None
    pixel_size = 1.0
    source_hint = ""

    # 1) TIFF
    if ext in (".tif", ".tiff"):
        try:
            from rsciio.tiff import file_reader as tiff_read
            tiff = tiff_read(path)
            if isinstance(tiff, list) and len(tiff) > 0 and "data" in tiff[0]:
                img = tiff[0]["data"]
                source_hint = "rsciio"

                # Best-effort pixel size
                try:
                    pixel_size = round(float(tiff[0]["axes"][0]["scale"]) * 1e6, 3)
                except Exception:
                    pixel_size = 1.0
        except Exception as e:
            print(f"RSCIIO Tiff Load failed: {e}. Falling back to OpenCV.")
            img = None

    # 2) DM3/DM4
    elif ext in (".dm3", ".dm4"):
        try:
            from rsciio.digitalmicrograph import file_reader as dm_read
            dm = dm_read(path)
            if isinstance(dm, list) and len(dm) > 0 and "data" in dm[0]:
                img = dm[0]["data"]
                source_hint = "rsciio"

                try:
                    pixel_size = round(
                        float(
                            dm[0]["original_metadata"]["ImageList"]["TagGroup0"]["ImageData"]
                            ["Calibrations"]["Dimension"]["TagGroup0"]["Scale"]
                        ),
                        4,
                    )
                except Exception:
                    pixel_size = 1.0
        except Exception as e:
            print(f"RSCIIO DM Load failed: {e}. Falling back to OpenCV.")
            img = None

    # 3) OpenCV fallback
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        source_hint = "cv2"

    if img is None:
        raise ValueError(f"CRITICAL ERROR: Could not load image data from {path}")

    # 4) Normalize to uint8 and enforce RGB(3ch)
    img_u8 = _to_uint8(img)
    img_rgb = _ensure_rgb(img_u8, source_hint=source_hint)

    return img_rgb.copy(), img, float(pixel_size)
