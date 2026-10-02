# -*- coding: utf-8 -*-
"""
SEM image calibration: OCR-based scale-bar detection and pixel-to-nanometer conversion.

Author: Karishma Begum
Year: 2026
"""

import cv2
import numpy as np
import re
import os
import glob
import pytesseract
from PIL import Image
from pathlib import Path

# Default template directory: <repo_root>/sem_templates
# (this file lives in <repo_root>/scripts/, so parent.parent is the repo root)
DEFAULT_TEMPLATE_DIR = str(Path(__file__).resolve().parent.parent / "sem_templates")

class SEMCalibrator:
    def __init__(self, template_dir=DEFAULT_TEMPLATE_DIR):
        """Initializes the calibrator and loads cached scale-bar text templates."""
        self.templates = {}
        self.template_dir = template_dir

        os.makedirs(self.template_dir, exist_ok=True)  # ensure it exists so we can save later

        all_files = glob.glob(os.path.join(self.template_dir, '*.png'))
        for f_path in all_files:
            filename = os.path.basename(f_path)
            if "temp_label" in filename:
                continue
            name = os.path.splitext(filename)[0]
            self.templates[name] = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)

    def _run_template_matching(self, gray_panel, w):
        """Matches the scale-bar text crop against cached templates, falling back to OCR on weak matches."""
        search_zone = gray_panel[14:44, int(w * 0.70):int(w * 0.90)]

        if not self.templates:
            return self._run_ocr_fallback(gray_panel, w, search_zone)

        # Estimate text width in the search zone (bright pixel span) to reject
        # templates whose text width doesn't match
        sz_bright_cols = np.where(np.max(search_zone, axis=0) > 140)[0]
        sz_text_width = (sz_bright_cols[-1] - sz_bright_cols[0] + 1) if len(sz_bright_cols) > 0 else search_zone.shape[1]

        best_score = -1
        second_best_score = -1
        detected_string = "Unknown"

        for name, tpl_img in self.templates.items():
            if tpl_img is None:
                continue

            th, tw = tpl_img.shape[:2]
            sh, sw = search_zone.shape[:2]

            # Skip templates whose text width differs too much from the observed width
            tpl_bright_cols = np.where(np.max(tpl_img, axis=0) > 140)[0]
            tpl_text_width = (tpl_bright_cols[-1] - tpl_bright_cols[0] + 1) if len(tpl_bright_cols) > 0 else tw
            width_ratio = tpl_text_width / max(sz_text_width, 1)
            if width_ratio < 0.6 or width_ratio > 1.6:
                continue

            # Resize template to fit the search zone without distorting aspect ratio
            if th > sh or tw > sw:
                scale = min(sh / th, sw / tw) * 0.95
                tpl_resized = cv2.resize(tpl_img, (max(1, int(tw * scale)), max(1, int(th * scale))))
            else:
                tpl_resized = tpl_img

            if tpl_resized.shape[0] > search_zone.shape[0] or tpl_resized.shape[1] > search_zone.shape[1]:
                continue

            res = cv2.matchTemplate(search_zone, tpl_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > best_score:
                second_best_score = best_score
                best_score = max_val
                detected_string = name
            elif max_val > second_best_score:
                second_best_score = max_val

        # Require both a high absolute score and a clear margin over the runner-up
        margin = best_score - max(second_best_score, 0)
        if best_score >= 0.82 and margin >= 0.05:
            numbers = re.findall(r'\d+\.?\d*', detected_string)
            value = float(numbers[0]) if numbers else 1.0
            unit_match = re.search(r'(nm|um|mm)', detected_string.lower())
            unit = unit_match.group() if unit_match else 'um'
            return value, unit

        return self._run_ocr_fallback(gray_panel, w, search_zone)

    def _run_ocr_fallback(self, gray_panel, w, search_zone=None):
        """Runs OCR on the scale-bar text crop and caches the result as a new template."""
        text_zone = gray_panel[14:44, int(w * 0.70):int(w * 0.90)]
        th, tw = text_zone.shape[:2]

        upscaled = cv2.resize(text_zone, (tw * 6, th * 6), interpolation=cv2.INTER_LANCZOS4)
        _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_text_zone = cv2.bitwise_not(thresh)

        config_options = ['--psm 6 --oem 3', '--psm 7 --oem 3']
        parsed_value, parsed_unit = 1.0, 'um'
        recognized = False

        for cfg in config_options:
            raw_ocr_text = pytesseract.image_to_string(processed_text_zone, config=cfg)
            cleaned_text = raw_ocr_text.strip().lower().replace('μ', 'um').replace('µ', 'um').replace('nn', 'nm').replace('mn', 'mm')

            numbers = re.findall(r'\d+\.?\d*', cleaned_text)
            unit_match = re.search(r'(nm|um|mm)', cleaned_text)

            if numbers:
                parsed_value = float(numbers[0])
                if unit_match:
                    parsed_unit = unit_match.group()
                recognized = True
                break

        if recognized and text_zone is not None:
            self._save_new_template(text_zone, parsed_value, parsed_unit)

        return parsed_value, parsed_unit

    def _save_new_template(self, text_zone_gray, value, unit):
        """Saves an OCR-recognized label crop as a template PNG for future direct matching."""
        # Format value without trailing ".0" for whole numbers, e.g. "1um" not "1.0um"
        value_str = f"{value:g}"
        name = f"{value_str}{unit}"

        # Avoid overwriting/duplicating an existing template
        if name in self.templates:
            print(f"[SEMCalibrator] Template '{name}' already in memory, skipping save.")
            return

        # Verify the directory exists and is writable before saving
        if not os.path.isdir(self.template_dir):
            print(f"[SEMCalibrator] WARNING: template_dir does not exist: {self.template_dir}")
            try:
                os.makedirs(self.template_dir, exist_ok=True)
                print(f"[SEMCalibrator] Created template_dir: {self.template_dir}")
            except Exception as e:
                print(f"[SEMCalibrator] ERROR: could not create template_dir: {e}")
                return

        if not os.access(self.template_dir, os.W_OK):
            print(f"[SEMCalibrator] ERROR: template_dir is not writable: {self.template_dir}")
            return

        save_path = os.path.join(self.template_dir, f"{name}.png")

        # Don't overwrite an existing file on disk that wasn't loaded into memory
        if os.path.exists(save_path):
            print(f"[SEMCalibrator] Template file already exists on disk: {save_path}")
        else:
            success = cv2.imwrite(save_path, text_zone_gray)
            if success:
                print(f"[SEMCalibrator] Saved new template: {save_path}")
            else:
                print(f"[SEMCalibrator] ERROR: cv2.imwrite failed for: {save_path}")
                print(f"[SEMCalibrator]   -> check that the path is valid and the array is a proper image "
                      f"(dtype={text_zone_gray.dtype}, shape={text_zone_gray.shape})")
                return

        # Register in memory for use within this session
        self.templates[name] = text_zone_gray.copy()

    def get_calibration(self, img_rgb, fallback_ratio=1.0):
        """Detects the scale bar in the image's info panel and computes the nm-per-pixel ratio."""
        h, w = img_rgb.shape[:2]
        arr = img_rgb

        # ── STEP 1: Detect the black info box ────────────────────────────────────
        row_avgs = [float(np.mean(arr[y])) for y in range(h)]
        bottom_border = None
        for y in range(h - 1, 0, -1):
            if row_avgs[y] > 180:
                bottom_border = y
                break
        if bottom_border is None: return fallback_ratio, "nm", arr[int(h * 0.92):h, :].copy()

        top_border = None
        in_dark = False
        for y in range(bottom_border - 1, 0, -1):
            if row_avgs[y] < 20: in_dark = True
            elif in_dark and row_avgs[y] > 150:
                top_border = y
                break
        if top_border is None: return fallback_ratio, "nm", arr[int(h * 0.92):h, :].copy()

        box_y_start, box_y_end = top_border + 1, bottom_border - 1
        box_h = box_y_end - box_y_start + 1

        # ── STEP 2: Find the scale bar panel boundary ──────────────────────────────
        box_interior = arr[box_y_start:box_y_end + 1, :]
        gray_box = np.mean(box_interior, axis=2)
        col_counts = np.sum(gray_box > 150, axis=0)
        full_h_cols = np.where(col_counts >= box_h * 0.85)[0]
        x_split = int(full_h_cols[full_h_cols < int(w * 0.75)][-1]) + 1 if len(full_h_cols) > 0 else int(w * 0.50)

        # ── STEP 3: Find the scale bar row ────────────────────────────────────────
        best_max_cl, scale_row = 0, None
        for y in range(box_y_start, box_y_end + 1):
            row = np.mean(arr[y, x_split:], axis=1)
            bright = np.where(row > 150)[0]
            if len(bright) < 3 or len(bright) > (w - x_split) * 0.85: continue
            cl = []
            s = p = int(bright[0])
            for c in bright[1:]:
                if int(c) - p > 5:
                    cl.append(p - s + 1)
                    s = int(c)
                p = int(c)
            cl.append(p - s + 1)
            if max(cl) > best_max_cl:
                best_max_cl, scale_row = max(cl), y

        if scale_row is None: return fallback_ratio, "nm", arr[top_border:h, :].copy()

        # ── STEP 4: Smart Endpoint Logic ──────────────────────────────────────────
        row_full = np.mean(arr[scale_row], axis=1)
        panel_bright = np.where(row_full > 140)[0]
        panel_bright = panel_bright[panel_bright >= x_split]
        if len(panel_bright) == 0: return fallback_ratio, "nm", arr[top_border:h, :].copy()

        clusters = []
        s = p = int(panel_bright[0])
        for c in panel_bright[1:]:
            if int(c) - p > 20:
                clusters.append((s, p))
                s = int(c)
            p = int(c)
        clusters.append((s, p))
        main_cluster = max(clusters, key=lambda x: x[1] - x[0])
        abs_x_start, abs_x_end = int(main_cluster[0]), int(main_cluster[1]) - 3
        pixel_length = abs_x_end - abs_x_start + 1

        # ── STEP 5: Template Matching Logic ─────────────────────────────────────
        panel_crop = arr[int(h * 0.92):h, :]
        gray_panel = cv2.cvtColor(panel_crop, cv2.COLOR_RGB2GRAY)

        best_val, best_unit = self._run_template_matching(gray_panel, w)

        # ── MATH & NORMALIZATION UNIT ─────────────────────────────────────────────
        val_in_nm = best_val * 1000 if best_unit == 'um' else (best_val * 1000000 if best_unit == 'mm' else best_val)
        final_ratio = val_in_nm / pixel_length

        # ── STEP 6: Annotation Rendering ──────────────────────────────────────────
        roi_shift_y = max(0, top_border - 20)
        label_roi = arr[roi_shift_y:h, :].copy()
        local_y = scale_row - roi_shift_y

        cv2.line(label_roi, (abs_x_start, local_y), (abs_x_end - 1, local_y), (255, 0, 0), 1)
        cv2.circle(label_roi, (abs_x_start, local_y), 3, (255, 0, 0), -1)
        cv2.circle(label_roi, (abs_x_end, local_y), 3, (255, 0, 0), -1)

        label_text = f"{pixel_length}px | {best_val}{best_unit} | {best_val/pixel_length:.4f}{best_unit}/px"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(label_roi, (abs_x_start - 3, local_y - 14 - text_size[1] - 3), (abs_x_start + text_size[0] + 3, local_y - 14 + 3), (0, 0, 0), -1)
        cv2.putText(label_roi, label_text, (abs_x_start, local_y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

        return final_ratio, "nm", label_roi
