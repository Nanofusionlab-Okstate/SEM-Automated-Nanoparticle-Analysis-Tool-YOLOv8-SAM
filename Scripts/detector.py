# -*- coding: utf-8 -*-
"""
Updated: Oct 2026 
@author: Karishma Begum 
"""
import os
import numpy as np
import cv2
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------
# Root setup
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_CKPT = PROJECT_ROOT / "best12x.pt"
VIT_H_CKPT = PROJECT_ROOT / "sam_vit_h_4b8939.pth"
MOBILE_SAM_CKPT = PROJECT_ROOT / "mobile_sam.pt"

def _pick_torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def find_label_crop_row(img_rgb, scan_bottom_frac=0.12, dark_thresh=35, min_dark_ratio=0.90, min_label_rows_frac=0.15):
    """
    Detects the black footer bar typical in SEM images to exclude it from analysis.
    """
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
    """
    Replicates ImageJ's 'Process > Binary > Watershed'.
    Uses the distance transform to find 'peaks' and cuts narrow bridges/joints.
    """
    mask_8u = (mask_binary * 255).astype(np.uint8)
    
    # 1. Distance Transform
    dist_transform = cv2.distanceTransform(mask_8u, cv2.DIST_L2, 5)
    
    # 2. Threshold to find centers 
    # 0.4 * max is a balanced setting for nanoparticle clusters
    _, markers_img = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0) 
    markers_img = np.uint8(markers_img)
    
    # 3. Label markers
    num_labels, markers = cv2.connectedComponents(markers_img)
    
    if num_labels <= 2: # Only background + 1 object
        return [mask_binary]
        
    # 4. Watershed
    rgb_mask = cv2.cvtColor(mask_8u, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(rgb_mask, markers)
    
    split_results = []
    for label in range(2, num_labels + 1):
        m = np.zeros_like(mask_binary)
        m[markers == label] = 1
        if np.sum(m) > 10:
            split_results.append(m)
            
    return split_results

# ---------------------------
#  HYBRID DETECTION
# ---------------------------
def object_detection(
    img_x,
    img_meta=None,
    path_to_image=None,
    img_size=1024,
    pred_score=0.25,
    overlap_thr=0.25,
    save=False,
    s_txt=False,
):
    from ultralytics import YOLO
    device = _pick_torch_device()

    if img_x is None:
        return np.array([]), [], None, None, False

    # --- 1) PRE-PROCESSING & CROPPING ---
    crop_row, label_found = find_label_crop_row(img_x)
    analysis_img = img_x[:crop_row, :] if label_found else img_x
    h_adj, w_adj = analysis_img.shape[:2]
    
    if analysis_img.ndim == 2:
        analysis_img = cv2.cvtColor(analysis_img, cv2.COLOR_GRAY2RGB)
    elif analysis_img.shape[2] == 4:
        analysis_img = analysis_img[:, :, :3]

    gray_for_rescue = cv2.cvtColor(analysis_img, cv2.COLOR_RGB2GRAY)
    all_raw_masks = []
    all_raw_bboxes = []

    # Initialize a clean canvas for your consolidated bounding box panel
    bbox_panel = cv2.cvtColor(analysis_img.copy(), cv2.COLOR_RGB2BGR)

    # --- 2) PRIMARY YOLO + SAM DETECTION (MULTI-SCALE & ASPECT PRESERVING) ---
    model = YOLO(str(YOLO_CKPT))
    
    scales = [img_size, 640] if img_size > 640 else [img_size]
    all_yolo_bboxes = []

    # Predict across scales with rect=True to prevent aspect ratio distortion
    for sz in scales:
        results = model.predict(
            source=analysis_img, 
            imgsz=sz, 
            conf=pred_score, 
            iou=overlap_thr, 
            rect=True, 
            agnostic_nms=True, 
            device=device, 
            verbose=False
        )
        if results[0].boxes:
            all_yolo_bboxes.extend(results[0].boxes.xyxy.tolist())

    # Deduplicate candidate bounding boxes across scales via NMS
    if len(all_yolo_bboxes) > 0:
        boxes_tensor = torch.tensor(all_yolo_bboxes, device=device)
        scores = torch.ones(len(boxes_tensor), device=device)
        keep_indices = torch.ops.torchvision.nms(boxes_tensor, scores, iou_threshold=0.35)
        yolo_bboxes = boxes_tensor[keep_indices].cpu().numpy().tolist()
    else:
        yolo_bboxes = []

    # Draw primary YOLO bounding boxes onto the panel canvas
    if len(yolo_bboxes) > 0:
        for box in yolo_bboxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(bbox_panel, (x1, y1), (x2, y2), (0, 255, 0), 2) # Vibrant Green Outline

        sam = sam_model_registry["vit_h"](checkpoint=str(VIT_H_CKPT))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(analysis_img)

        input_boxes = torch.tensor(yolo_bboxes, device=device, dtype=torch.float32)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, (h_adj, w_adj))

        masks_torch, _, _ = predictor.predict_torch(None, None, boxes=transformed_boxes, multimask_output=False)
        masks_yolo = masks_torch.detach().cpu().numpy().squeeze()
        
        if masks_yolo.ndim == 2: 
            masks_yolo = np.expand_dims(masks_yolo, axis=0)

        for i, m in enumerate(masks_yolo):
            all_raw_masks.append((m > 0).astype("uint8"))
            all_raw_bboxes.append(yolo_bboxes[i])

    # --- 3) Adaptive Threshold (Rescue) ---
    # Scale block size dynamically relative to width to prevent large particles from fragmenting
    dynamic_block = max(35, int(w_adj / 20) | 1)

    thresh = cv2.adaptiveThreshold(
        gray_for_rescue, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        dynamic_block, 
        5
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 10000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                is_duplicate = False
                for bx in yolo_bboxes:
                    if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    m_res = np.zeros((h_adj, w_adj), dtype=np.uint8)
                    cv2.drawContours(m_res, [cnt], -1, 1, -1)
                    all_raw_masks.append(m_res)
                    x, y, w, h = cv2.boundingRect(cnt)
                    all_raw_bboxes.append([float(x), float(y), float(x+w), float(y+h)])
                    
                    # Force bounding box around rescued adaptive threshold instances
                    cv2.rectangle(bbox_panel, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Save the complete consolidated bounding box panel into your central output workspace folder
    if path_to_image:
        orig_name = Path(path_to_image).stem
        save_dir = PROJECT_ROOT / "outputs" / orig_name
        save_dir.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_dir / f"{orig_name}_yolo_boxes.png"), bbox_panel)

    # --- 4) WATERSHED SPLIT & SCIENTIFIC FILTERING ---
    filtered_masks = []
    filtered_bboxes = []

    for mask_binary in all_raw_masks:
        # Split merged particles using Watershed
        potential_splits = apply_watershed_split(mask_binary)
        
        for sm in potential_splits:
            area = int(np.sum(sm))

            # Filter 1: Size 
            if area < 20: 
                continue

            cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            cnt = max(cnts, key=cv2.contourArea)
            
            perim = cv2.arcLength(cnt, True)
            circularity = (4.0 * np.pi * area) / (perim * perim + 1e-6)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = max(bw, bh) / max(1, min(bw, bh))

            # Filter 2: Shape Check 
            if circularity > 0.80 and aspect_ratio < 9:  
                filtered_masks.append(sm)
                filtered_bboxes.append([float(x), float(y), float(x+bw), float(y+bh)])

    # --- 5) OUTPUT PACKAGING ---
    if not filtered_masks:
        return np.array([]), [], analysis_img, crop_row, label_found

    final_masks_stack = np.stack(filtered_masks)

    return final_masks_stack, filtered_bboxes, analysis_img, crop_row, label_found
