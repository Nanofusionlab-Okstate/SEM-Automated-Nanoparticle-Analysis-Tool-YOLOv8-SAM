# -*- coding: utf-8 -*-
"""
Image-quality checks for nanoparticle detection viability.

Author: Karishma Begum
Year: 2026
"""


import cv2
import numpy as np

def check_image_viability(img_rgb):
    """
    Checks nanoparticle image viability based on particle circularity and solidity.

    Returns:
        (bool, str, dict): pass/fail, a message, and a score/metrics dict.
    """
    if img_rgb is None:
        return False, "Invalid image data", {"score": 0}
    
    # 1. Configuration Constants (Mirrored from test logic)
    MIN_CIRCULARITY = 0.75  
    MIN_SOLIDITY = 0.90     
    PASS_THRESHOLD = 0.70  
    MIN_AREA = 40  # Threshold for noise

    # 2. Prepare Image
    # Note: Using RGB to Gray conversion as the tool likely passes RGB frames
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Shape Analysis Metrics
    valid_nps = 0
    total_blobs = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA: 
            continue # Ignore tiny noise
        
        total_blobs += 1
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: 
            continue
        
        # Metric: Circularity (4*pi*A / P^2)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Metric: Solidity (Area / Convex Hull Area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        if circularity > MIN_CIRCULARITY and solidity > MIN_SOLIDITY:
            valid_nps += 1

    # 4. Calculation & Validation Gates
    quality_ratio = valid_nps / total_blobs if total_blobs > 0 else 0
    
    # Check A: Sparsity Gate
    if total_blobs < 10:
        return (
            False,
            f"Insufficient particle density ({total_blobs} resolvable regions detected)",
            {"score": round(quality_ratio, 2), "count": total_blobs}
        )
    
    # Check B: Quality Gate
    if quality_ratio < PASS_THRESHOLD:
        return (
            False,
            f"Low proportion of well-resolved particles (quality ratio: {quality_ratio:.2f}); "
            f"micrograph dominated by elongated or incompletely resolved regions",
            {"score": round(quality_ratio, 2), "count": total_blobs}
        )

    # Check C: Bulk Mass Gate (safety check for large clumps)
    h, w = gray.shape
    total_pixels = h * w
    max_blob_area = max([cv2.contourArea(c) for c in contours]) if contours else 0
    if (max_blob_area / total_pixels) > 0.35:
        return (
            False,
            "Micrograph dominated by an unformed, continuous bulk region rather than discrete particles",
            {"score": round(quality_ratio, 2), "mass_ratio": round(max_blob_area/total_pixels, 2)}
        )

    return True, f"Valid: quality ratio {quality_ratio:.2f}", {"score": round(quality_ratio, 2), "count": valid_nps}
