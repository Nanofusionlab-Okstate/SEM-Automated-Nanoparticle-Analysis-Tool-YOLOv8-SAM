# -*- coding: utf-8 -*-
"""
Perspective-mode detection pipeline for tilted/oblique SEM micrographs: extracts
particle contact angle, height, and base diameter via spherical-cap geometry.

Author: Karishma Begum
Year: 2026
"""

import os
import math
import numpy as np
import cv2
import torch
import pandas as pd
from scipy.spatial import KDTree
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------
# Root setup
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_CKPT = PROJECT_ROOT / "best12x.pt"
VIT_H_CKPT = PROJECT_ROOT / "sam_vit_h_4b8939.pth"

def _pick_torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def find_label_crop_row(img_rgb, scan_bottom_frac=0.12, dark_thresh=35, min_dark_ratio=0.90, min_label_rows_frac=0.15):
    """Detects and crops the black SEM metadata footer."""
    if img_rgb is None: return None, False
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb.copy()
    h, w = gray.shape[:2]
    y0 = int(h * (1.0 - scan_bottom_frac))
    bottom = gray[y0:, :]
    dark_ratio_per_row = (bottom < dark_thresh).mean(axis=1)
    label_found = (dark_ratio_per_row > min_dark_ratio).mean() > min_label_rows_frac

    if not label_found: return h, False

    run = 0
    run_needed = max(12, int(0.02 * h))
    for i, r in enumerate(dark_ratio_per_row):
        if r > min_dark_ratio:
            run += 1
            if run >= run_needed:
                return int(np.clip(y0 + i - run_needed + 1, 0, h)), True
        else: run = 0
    return y0, True

def apply_watershed_split(mask_binary):
    """Splits connected particles using distance transform and watershed separation."""
    mask_8u = (mask_binary * 255).astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_8u, cv2.DIST_L2, 5)
    # Distance-transform peaks above 35% of max seed the watershed markers.
    _, markers_img = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
    markers_img = np.uint8(markers_img)

    num_labels, markers = cv2.connectedComponents(markers_img)
    if num_labels <= 2:
        return [mask_binary]

    rgb_mask = cv2.cvtColor(mask_8u, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(rgb_mask, markers)

    split_results = []
    for label in range(2, num_labels + 1):
        m = np.zeros_like(mask_binary)
        m[markers == label] = 1
        if np.sum(m) > 10:  # drop slivers under 10px
            split_results.append(m)

    return split_results

# ---------------------------
#  SPACING & FFT CALCULATORS
# ---------------------------
def compute_nearest_neighbor_spacing(centroids, nm_per_pixel=1.0):
    """Calculates mean nearest-neighbor distance (NND) between particle centroids."""
    if len(centroids) < 2:
        return 0.0
    tree = KDTree(centroids)
    distances, _ = tree.query(centroids, k=2)  # k=2 because k=1 is the point itself
    nnd_px = distances[:, 1]
    return float(np.mean(nnd_px) * nm_per_pixel)

def compute_fft_spacing(gray_img, nm_per_pixel=1.0):
    """Computes global structural periodicity using 2D Fast Fourier Transform."""
    if gray_img is None or gray_img.size == 0:
        return 0.0

    h, w = gray_img.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fft = np.fft.fftshift(np.fft.fft2(gray_img * window))
    magnitude_spectrum = np.abs(fft) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r = np.hypot(x, y).astype(int)

    radial_sum = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    radial_count = np.bincount(r.ravel())
    radial_profile = radial_sum / np.maximum(radial_count, 1)

    if len(radial_profile) < 10:
        return 0.0

    radial_profile[:5] = 0
    peak_r = np.argmax(radial_profile)

    if peak_r == 0:
        return 0.0

    dominant_period_px = max(h, w) / peak_r
    return float(dominant_period_px * nm_per_pixel)

# ---------------------------
#  SHAPE / OVERLAP HELPERS
# ---------------------------
def _compute_solidity(cnt):
    """Solidity = contour area / convex-hull area; low values flag crescents or occluded fragments."""
    area = cv2.contourArea(cnt)
    if area <= 0:
        return 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area) / float(hull_area)

def _boxes_iou(box_a, box_b):
    """Standard IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

def _mask_overlap_ratio(mask_a, mask_b):
    """Overlap fraction relative to the smaller mask -- used to detect occlusion between stacked particles."""
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    smaller = min(a.sum(), b.sum())
    return float(inter) / float(smaller) if smaller > 0 else 0.0

# ---------------------------
#  ILLUMINATION / SCALE HELPERS
# ---------------------------
def _adaptive_clahe_clip(gray_img, base_clip_limit):
    """Scales CLAHE clip limit down for already high-contrast images and up to full strength for low-contrast ones."""
    p2, p98 = np.percentile(gray_img, [2, 98])
    spread = float(p98 - p2)  # 0-255 scale; low spread = genuinely low-contrast image
    if spread >= 180:
        scale = 0.15   # already well-exposed -- only a token amount of enhancement
    elif spread <= 60:
        scale = 1.0     # genuinely low-contrast/low-light -- full strength
    else:
        scale = 1.0 - (spread - 60) / (180 - 60) * 0.85
    return max(0.5, base_clip_limit * scale)

def _clahe_enhance(gray_img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Applies CLAHE local contrast normalization. Used only for the detection input, not for measurement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)

def _round_to_stride(x, stride=32):
    return int(np.ceil(x / stride) * stride)

# ---------------------------
#  CONTACT ANGLE GEOMETRY
# ---------------------------
def compute_particle_contact_angle(mask_binary, gray_img=None):
    """Computes contact angle, height, and base diameter from a particle mask via spherical-cap geometry."""
    cnts, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    if area < 15:
        return None

    ys = cnt[:, 0, 1]
    max_y = np.max(ys)
    min_y = np.min(ys)
    h = float(max_y - min_y)

    if h < 5:
        return None

    bottom_pts = cnt[ys >= (max_y - 3)]
    if len(bottom_pts) < 2:
        return None

    left_pt = bottom_pts[np.argmin(bottom_pts[:, 0, 0])][0]
    right_pt = bottom_pts[np.argmax(bottom_pts[:, 0, 0])][0]

    d = float(abs(right_pt[0] - left_pt[0]))
    if d < 5:
        return None

    apex_pt = cnt[np.argmin(ys)][0]
    baseline_y = int((left_pt[1] + right_pt[1]) / 2.0)

    rad_angle = 2.0 * math.atan((2.0 * h) / d)
    contact_angle_deg = math.degrees(rad_angle)

    M = cv2.moments(cnt)
    cx = M["m10"] / M["m00"] if M["m00"] != 0 else left_pt[0]
    cy = M["m01"] / M["m00"] if M["m00"] != 0 else max_y

    return {
        "contact_angle": round(contact_angle_deg, 1),
        "height_px": round(h, 2),
        "diameter_px": round(d, 2),
        "left_pt": (int(left_pt[0]), int(left_pt[1])),
        "right_pt": (int(right_pt[0]), int(right_pt[1])),
        "apex_pt": (int(apex_pt[0]), int(apex_pt[1])),
        "baseline_y": int(baseline_y),
        "centroid": (cx, cy)
    }

# ---------------------------
#  HYBRID PERSPECTIVE DETECTION
# ---------------------------
def perspective_detection(
    img_x,
    img_meta=None,
    path_to_image=None,
    img_size=None,                 # None = auto-sized to the input resolution (see step 2)
    pred_score=0.15,
    overlap_thr=0.35,
    nm_per_pixel=1.0,
    save=True,
    rescue_iou_thresh=0.15,        # dedup rescue boxes against YOLO boxes
    min_solidity=0.85,             # rejects crescent / jagged / occluded fragments
    occlusion_overlap_thresh=0.25, # greedy front/back resolution between final masks
    exclude_border=True,           # drop particles clipped by the frame edge
    clahe_clip_limit=3.0,          # contrast normalization strength for detection input
    recall_conf_scale=0.5,         # second YOLO pass runs at pred_score * this, for faint particles
    min_solidity_untrusted=0.92,   # stricter solidity bar for recall/rescue-sourced masks
    max_area_ratio_to_median=2.5,  # rejects masks far larger than this image's typical particle
    min_sam_confidence=0.80,       # SAM's own predicted mask-quality score, primary-source floor
    min_sam_confidence_untrusted=0.88,  # stricter SAM-confidence floor for recall/rescue sources
    **kwargs
):
    """Detects particles in an oblique SEM image and computes per-particle contact-angle geometry, NND, and FFT spacing."""
    pred_score = kwargs.get('conf_thresh', pred_score)
    nm_per_pixel = kwargs.get('nm_per_pixel', nm_per_pixel)

    device = _pick_torch_device()
    if img_x is None:
        return np.array([]), [], [], None, None, False, None, 0.0, 0.0

    # --- 1) PRE-PROCESSING & CROPPING ---
    crop_row, label_found = find_label_crop_row(img_x)
    analysis_img = img_x[:crop_row, :] if label_found else img_x
    h_adj, w_adj = analysis_img.shape[:2]

    if analysis_img.ndim == 2:
        analysis_img = cv2.cvtColor(analysis_img, cv2.COLOR_GRAY2RGB)
    elif analysis_img.shape[2] == 4:
        analysis_img = analysis_img[:, :, :3]

    gray_for_rescue = cv2.cvtColor(analysis_img, cv2.COLOR_RGB2GRAY)
    contact_panel = cv2.cvtColor(analysis_img.copy(), cv2.COLOR_RGB2BGR)

    # --- 2) DETECTION INPUT: CONTRAST-NORMALIZED, RESOLUTION-ADAPTIVE ---
    # CLAHE-normalize contrast so YOLO/SAM see a consistent profile regardless
    # of exposure. Measurements still use the original, unmodified pixels.
    effective_clahe_clip = _adaptive_clahe_clip(gray_for_rescue, clahe_clip_limit)
    gray_enhanced = _clahe_enhance(gray_for_rescue, clip_limit=effective_clahe_clip)
    detect_img = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2RGB)

    # Auto-size the YOLO input to the image's own resolution (clamped, stride-32).
    if img_size is None:
        img_size = _round_to_stride(int(np.clip(max(h_adj, w_adj), 640, 1536)))

    # --- 3) PRIMARY + RECALL YOLO PASSES (boxes only -- SAM runs once, later) ---
    # A second, lower-confidence pass recovers particles missed under weak
    # contrast; every candidate still goes through SAM + shape filtering below.
    from ultralytics import YOLO
    model = YOLO(str(YOLO_CKPT))

    def _yolo_boxes(conf):
        res = model.predict(
            source=detect_img, imgsz=img_size, conf=conf, iou=overlap_thr,
            agnostic_nms=True, device=device, verbose=False
        )
        return res[0].boxes.xyxy.tolist() if res[0].boxes else []

    primary_boxes = _yolo_boxes(pred_score)
    recall_conf = max(0.03, pred_score * recall_conf_scale)
    recall_boxes_raw = _yolo_boxes(recall_conf)
    recall_boxes = [
        b for b in recall_boxes_raw
        if not any(_boxes_iou(b, pb) > 0.3 for pb in primary_boxes)
    ]
    yolo_bboxes = primary_boxes + recall_boxes

    # --- 4) ADAPTIVE THRESHOLD RESCUE -- CANDIDATE BOXES ONLY ---
    # Proposes boxes only; every accepted box is segmented by SAM in the
    # unified pass below so all masks share the same smooth shape quality.
    blurred_gray = cv2.bilateralFilter(gray_enhanced, 5, 50, 50)
    dynamic_block = max(35, int(w_adj / 20) | 1)
    thresh = cv2.adaptiveThreshold(
        blurred_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        dynamic_block,
        5
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rescue area bounds are derived from the median size of YOLO's own
    # detections in this image, so they scale with resolution/magnification.
    if primary_boxes:
        box_areas = [max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in primary_boxes]
        median_box_area = float(np.median(box_areas))
        rescue_area_lo = max(40.0, 0.15 * median_box_area)
        rescue_area_hi = max(rescue_area_lo * 2.0, 3.0 * median_box_area)
    else:
        median_box_area = 900.0
        rescue_area_lo, rescue_area_hi = 80.0, 8000.0

    rescue_bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        solidity = _compute_solidity(cnt)

        # Drop tiny slivers, non-round blobs, and non-convex fragments.
        if not (rescue_area_lo < area < rescue_area_hi and circularity > 0.50 and solidity > min_solidity):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cand_box = [float(x), float(y), float(x + w), float(y + h)]

        # Dedup against YOLO by box IoU rather than centroid-in-box.
        is_duplicate = any(_boxes_iou(cand_box, yb) > rescue_iou_thresh for yb in yolo_bboxes)
        if not is_duplicate:
            rescue_bboxes.append(cand_box)

    # --- 5) SINGLE UNIFIED SAM PASS ---
    # Every box (YOLO + surviving rescue candidates) is segmented together so
    # all masks share consistent, smooth edge quality. Each box keeps a
    # "source" tag; recall/rescue boxes get stricter shape thresholds below
    # since they come from a deliberately loosened detection pass.
    box_records = (
        [{"box": b, "source": "primary"} for b in primary_boxes]
        + [{"box": b, "source": "recall"} for b in recall_boxes]
        + [{"box": b, "source": "rescue"} for b in rescue_bboxes]
    )
    all_boxes = [r["box"] for r in box_records]
    box_sources = [r["source"] for r in box_records]
    all_raw_masks = []

    # Raw candidate boxes before any filtering, for diagnostics.
    # Color-coded by source: green = primary YOLO, yellow = recall, magenta = rescue.
    yolo_panel = cv2.cvtColor(analysis_img.copy(), cv2.COLOR_RGB2BGR)
    _box_colors = {"primary": (0, 200, 0), "recall": (0, 220, 220), "rescue": (220, 0, 220)}
    for rec in box_records:
        bx1, by1, bx2, by2 = [int(v) for v in rec["box"]]
        cv2.rectangle(yolo_panel, (bx1, by1), (bx2, by2), _box_colors[rec["source"]], 1)

    if len(all_boxes) > 0:
        sam = sam_model_registry["vit_h"](checkpoint=str(VIT_H_CKPT))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(detect_img)

        input_boxes = torch.tensor(all_boxes, device=device, dtype=torch.float32)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, (h_adj, w_adj))
        # Capture SAM's own predicted mask-quality (IoU) score per box.
        masks_torch, iou_preds_torch, _ = predictor.predict_torch(None, None, boxes=transformed_boxes, multimask_output=False)
        masks_sam = masks_torch.detach().cpu().numpy().squeeze()
        iou_preds = iou_preds_torch.detach().cpu().numpy().reshape(-1)

        if masks_sam.ndim == 2:
            masks_sam = np.expand_dims(masks_sam, axis=0)

        all_raw_masks = list(zip(
            [(m > 0).astype("uint8") for m in masks_sam],
            box_sources,
            iou_preds
        ))

    # --- 6) WATERSHED SPLIT & GEOMETRY, COLLECTING CANDIDATES ---
    # Minimum candidate mask area, scaled to this image's typical particle size.
    min_candidate_area = max(25.0, 0.10 * median_box_area)

    candidates = []
    for mask_binary, src, sam_conf in all_raw_masks:
        # SAM confidence is per original box; watershed-split pieces inherit it.
        req_sam_conf = min_sam_confidence if src == "primary" else max(min_sam_confidence, min_sam_confidence_untrusted)
        if sam_conf < req_sam_conf:
            continue

        potential_splits = apply_watershed_split(mask_binary)
        for sm in potential_splits:
            if int(np.sum(sm)) < min_candidate_area:
                continue

            cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)

            cnt_area = cv2.contourArea(cnt)
            cnt_perim = cv2.arcLength(cnt, True)
            circularity = (4.0 * np.pi * cnt_area) / (cnt_perim ** 2) if cnt_perim > 0 else 0.0

            # Primary YOLO boxes keep a looser shape bar; recall/rescue need
            # to look convincingly like a round, convex particle.
            if src == "primary":
                req_solidity, req_circularity = min_solidity, 0.55
            else:
                req_solidity, req_circularity = max(min_solidity, min_solidity_untrusted), 0.60

            # Masks well above this image's typical particle size are likely
            # an under-split touching pair, so require near-full convexity.
            reference_particle_area = max(1.0, 0.7 * median_box_area)
            if (cnt_area / reference_particle_area) > 1.4:
                req_solidity = max(req_solidity, 0.93)

            # Reject crescents/partial slivers (occluded/rear particles).
            if _compute_solidity(cnt) < req_solidity:
                continue
            if circularity < req_circularity:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Reject elongated blobs spanning gaps between real particles.
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if aspect > 2.5:
                continue

            if exclude_border and (x <= 1 or y <= 1 or (x + bw) >= (w_adj - 1) or (y + bh) >= (h_adj - 1)):
                continue

            geo_info = compute_particle_contact_angle(sm, gray_img=gray_for_rescue)
            if geo_info is None:
                continue
            # Contact angle is physically bounded to [0, 180] degrees.
            if not (5 <= geo_info["contact_angle"] <= 180):
                continue
            if geo_info["diameter_px"] < 10 or geo_info["height_px"] < 6:
                continue
            if geo_info["baseline_y"] >= (h_adj - 10):
                continue

            candidates.append({
                "mask": sm,
                "area": int(np.sum(sm)),
                "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                "geo": geo_info,
            })

    # --- 6b) SIZE-OUTLIER REJECTION (BEFORE OCCLUSION RESOLUTION) ---
    # Runs before occlusion resolution so an oversized merged-blob false
    # positive can't be kept first and wrongly reject real particles under it.
    if len(candidates) >= 6:
        areas = np.array([c["area"] for c in candidates], dtype=float)
        median_area = float(np.median(areas))
        if median_area > 0:
            candidates = [c for c in candidates if c["area"] <= max_area_ratio_to_median * median_area]

    # --- 7) OCCLUSION RESOLUTION: KEEP THE FRONT PARTICLE OF ANY STACK ---
    # Accept candidates largest-first; reject anything substantially
    # overlapping a particle already kept (the occluded/rear one).
    candidates.sort(key=lambda c: c["area"], reverse=True)

    filtered_masks, filtered_bboxes, contact_angle_data, centroids = [], [], [], []
    for cand in candidates:
        overlaps_existing = any(
            _mask_overlap_ratio(cand["mask"], kept) > occlusion_overlap_thresh
            for kept in filtered_masks
        )
        if overlaps_existing:
            continue

        filtered_masks.append(cand["mask"])
        filtered_bboxes.append(cand["bbox"])
        contact_angle_data.append(cand["geo"])
        centroids.append(cand["geo"]["centroid"])

        geo_info = cand["geo"]
        l_pt, r_pt, a_pt, base_y = geo_info["left_pt"], geo_info["right_pt"], geo_info["apex_pt"], geo_info["baseline_y"]
        x, y = int(cand["bbox"][0]), int(cand["bbox"][1])

        cv2.line(contact_panel, l_pt, r_pt, (0, 0, 255), 1)
        cv2.line(contact_panel, (a_pt[0], a_pt[1]), (a_pt[0], base_y), (255, 255, 0), 1)
        cv2.putText(
            contact_panel,
            # "°" has no Hershey-font glyph in OpenCV, so the unit is omitted here.
            f"{int(round(geo_info['contact_angle']))}",
            (x, max(10, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # --- 8) OUTPUT SAVING TO DISK ---
    if path_to_image and save:
        orig_name = Path(path_to_image).stem
        save_dir = PROJECT_ROOT / "outputs" / orig_name
        save_dir.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_dir / f"{orig_name}_contact_angles.png"), contact_panel)
        cv2.imwrite(str(save_dir / f"{orig_name}_yolo_detections.png"), yolo_panel)

    # --- 9) CALCULATE SPACING AND FFT METRICS ---
    avg_spacing_nnd = compute_nearest_neighbor_spacing(centroids, nm_per_pixel=nm_per_pixel)
    global_fft_spacing = compute_fft_spacing(gray_for_rescue, nm_per_pixel=nm_per_pixel)

    if not filtered_masks:
        return np.array([]), [], [], analysis_img, crop_row, label_found, contact_panel, 0.0, 0.0

    final_masks_stack = np.stack(filtered_masks)

    return (
        final_masks_stack,
        filtered_bboxes,
        contact_angle_data,
        analysis_img,
        crop_row,
        label_found,
        contact_panel,
        avg_spacing_nnd,
        global_fft_spacing
    )

def build_perspective_figures(df, user_x_limit=None, user_y_limit=None, unit="nm", **kwargs):
    """
    Builds Plotly dashboards of h/d ratio and contact angle vs. particle diameter.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figs = {}
    if df is None or df.empty or 'Contact_Angle_deg' not in df.columns:
        return figs

    height_col = f'Height_{unit}' if f'Height_{unit}' in df.columns else ('Height_nm' if 'Height_nm' in df.columns else 'height_px')
    diameter_col = (
        'Eq_Diameter' if 'Eq_Diameter' in df.columns
        else (f'Diameter_{unit}' if f'Diameter_{unit}' in df.columns
              else ('Diameter_nm' if 'Diameter_nm' in df.columns else 'diameter_px'))
    )

    if height_col not in df.columns or diameter_col not in df.columns:
        return figs

    plot_df = df[[height_col, diameter_col, 'Contact_Angle_deg']].dropna()
    plot_df = plot_df[plot_df[diameter_col] > 0]
    if plot_df.empty:
        return figs

    hd_ratio = plot_df[height_col] / plot_df[diameter_col]
    diameter_data = plot_df[diameter_col]
    active_x_limit = user_x_limit if user_x_limit is not None and user_x_limit > 0 else float(np.ceil(diameter_data.max()))

    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("h/d Ratio vs Diameter", "Contact Angle vs Diameter")
    )

    fig1.add_trace(
        go.Scatter(x=diameter_data, y=hd_ratio, mode='markers', marker=dict(color='#2980b9'), name="h/d"),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=diameter_data, y=plot_df['Contact_Angle_deg'], mode='markers', marker=dict(color='#e74c3c'), name="Contact Angle"),
        row=1, col=2
    )

    fig1.update_xaxes(range=[0, active_x_limit], title_text=f"Diameter ({unit})", row=1, col=1)
    fig1.update_xaxes(range=[0, active_x_limit], title_text=f"Diameter ({unit})", row=1, col=2)
    fig1.update_yaxes(title_text="h/d Ratio", row=1, col=1)
    fig1.update_yaxes(title_text="Contact Angle (°)", row=1, col=2)
    fig1.update_layout(height=450, template="plotly_white", showlegend=False)

    figs['dist'] = fig1
    return figs
