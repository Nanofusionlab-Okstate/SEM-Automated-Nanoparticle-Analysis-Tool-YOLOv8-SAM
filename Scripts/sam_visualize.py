# -*- coding: utf-8 -*-
"""
Visualization utilities for rendering SAM segmentation masks and particle overlays.

Author: Karishma Begum
Year: 2026
"""

import numpy as np
import cv2
import seaborn as sns
from pathlib import Path
import matplotlib.cm as cm
import colorcet as cc  
import cmasher as cmr  
from PIL import Image, ImageDraw, ImageFont 

def draw_text_scientific(canvas, text, position, font_size, color=(255, 255, 255), anchor="left"):
    """Renders Unicode text (e.g. ± or °) onto an image array using PIL."""
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    # Try common Arial install locations across platforms, falling back to
    # the metrically-equivalent Liberation Sans if Arial isn't available.
    arial_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    font = None
    for candidate in arial_candidates:
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except Exception:
            continue

    if font is None:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size
            )
        except Exception:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size
                )
            except Exception:
                try:
                    font = ImageFont.load_default(size=font_size)
                except TypeError:
                    font = ImageFont.load_default()

    if anchor == "center":
        bbox = draw.textbbox((0, 0), text, font=font)
        w_text = bbox[2] - bbox[0]
        position = (int(canvas.shape[1] / 2 - w_text / 2), position[1])
    elif anchor == "right":
        bbox = draw.textbbox((0, 0), text, font=font)
        w_text = bbox[2] - bbox[0]
        position = (canvas.shape[1] - position[0] - w_text, position[1])

    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)

def detect_metric_column(df):
    """Detects primary metrics column across Standard and Perspective analysis dataframes."""
    if df is None or df.empty:
        return None, ""
    
    candidate_cols = [
        ('Contact_Angle_deg', '°'),
        ('Height_nm', 'nm'),
        ('Height_um', 'um'),
        ('Diameter_nm', 'nm'),
        ('Diameter_um', 'um'),
        ('Eq_Diameter', 'nm'),
        ('diameter_px', 'px'),
        ('height_px', 'px')
    ]
    
    for col, unit in candidate_cols:
        if col in df.columns:
            return col, unit
            
    return None, ""

def create_semantic_transformation(img, masks, df=None, split_ratio=0.5, crop_row=None, cmap_name='fire'):
    if img is None or masks is None or len(masks) == 0:
        return img

    if crop_row is not None and crop_row > 0:
        base_img = img[0:crop_row, :].copy()
    else:
        base_img = img.copy()
    
    h, w = base_img.shape[:2]
    split_line = int(w * split_ratio)
    
    metric_col, _ = detect_metric_column(df)
    
    if metric_col is not None and not df.empty:
        GLOBAL_MIN = max(0.0, df[metric_col].min() - 1.0)
        GLOBAL_MAX = df[metric_col].max() + 1.0
    else:
        GLOBAL_MIN = 0.0
        GLOBAL_MAX = 60.0
    
    is_random = (cmap_name == "random_vivid")
    cmap = None
    if not is_random:
        try:
            if hasattr(cc.cm, cmap_name):
                cmap = getattr(cc.cm, cmap_name)
            elif hasattr(cmr, cmap_name):
                cmap = getattr(cmr, cmap_name)
            else:
                cmap = cm.get_cmap(cmap_name)
        except Exception:
            cmap = cc.cm.fire

    texture_mask = base_img.astype(np.float32) / 255.0
    color_mask = np.ones_like(texture_mask)

    for i in range(len(masks)):
        mask_raw = masks[i]
        if mask_raw is None: continue
        
        mask = (np.asarray(mask_raw) > 0).astype(np.uint8)
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        if is_random:
            hue = int(360 * (i / max(1, len(masks))))
            hsv_pixel = np.uint8([[[hue // 2, 240, 245]]])
            color_rgb = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0] / 255.0
        else:
            if metric_col is not None and not df.empty:
                particle_id = i + 1
                match = df.loc[df['Particle_ID'] == particle_id, metric_col]
                if not match.empty:
                    val = match.values[0]
                    norm_val = np.clip((val - GLOBAL_MIN) / max(1e-5, (GLOBAL_MAX - GLOBAL_MIN)), 0.0, 1.0)
                else:
                    norm_val = 0.0 
            else:
                norm_val = 0.0 

            rgba_color = cmap(norm_val)
            color_rgb = np.array(rgba_color[:3]) 
        
        for c in range(3):
            color_mask[mask > 0, c] = color_rgb[c]

    colored_img = (texture_mask * color_mask * 255).astype(np.uint8)
    final_output = base_img.copy()
    final_output[:, split_line:] = colored_img[:, split_line:]
    
    return final_output

def highlight_particle(masks, img, particle_id, crop_row=None, highlight_color=(255, 0, 40),
                        outline_thickness=4, dim_others=True, dim_factor=0.35, zoom_pad_frac=0.6):
    """
    Dims every particle except `particle_id` and outlines that one in a
    bright color, so a selected Particle_ID can be located on the image.

    Args:
        masks: (N, H, W) binary mask stack; Particle_ID is 1-indexed
            (masks[particle_id - 1] is the mask for that row).
        img: the raw/original image, not the rendered segmentation PNG.
        crop_row: footer-crop row, consistent with the rest of the pipeline.

    Returns:
        (full_image_with_highlight, zoomed_crop_around_particle); either
        may be None if particle_id is invalid or masks are missing.
    """
    if img is None or masks is None or len(masks) == 0:
        return img, None

    if crop_row is not None and crop_row > 0:
        base_img = img[0:crop_row, :].copy()
    else:
        base_img = img.copy()

    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    elif base_img.shape[2] == 4:
        base_img = base_img[:, :, :3]

    h, w = base_img.shape[:2]

    masks_arr = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    if masks_arr.ndim == 2:
        masks_arr = masks_arr[None, ...]

    idx = int(particle_id) - 1
    if idx < 0 or idx >= masks_arr.shape[0]:
        return base_img, None

    target_raw = (masks_arr[idx] > 0).astype(np.uint8)
    if target_raw.shape[0] != h or target_raw.shape[1] != w:
        target_mask = cv2.resize(target_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        target_mask = target_raw

    if target_mask.sum() == 0:
        return base_img, None

    # Dim everything except the highlighted particle
    if dim_others:
        dimmed = (base_img.astype(np.float32) * dim_factor).astype(np.uint8)
        output = dimmed.copy()
        output[target_mask > 0] = base_img[target_mask > 0]
    else:
        output = base_img.copy()

    # Outline the particle
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(output, contours, -1, highlight_color, outline_thickness)

    # Bounding box and an arrow pointing to the centroid
    ys, xs = np.where(target_mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    arrow_len = max(25, int(0.03 * max(h, w)))
    cv2.arrowedLine(output, (cx - arrow_len, cy - arrow_len), (cx - 4, cy - 4),
                     highlight_color, 3, tipLength=0.35)
    label_pos = (max(0, cx - arrow_len - 10), max(20, cy - arrow_len - 12))
    output = draw_text_scientific(
        output, f"ID {particle_id}", label_pos, max(18, int(0.02 * max(h, w))),
        color=highlight_color
    )

    # Zoomed-in crop around the particle
    bw, bh = (x1 - x0), (y1 - y0)
    pad = int(max(bw, bh, 10) * zoom_pad_frac) + 10
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
    crop_img = output[cy0:cy1, cx0:cx1].copy()
    if crop_img.size == 0:
        crop_img = None

    return output, crop_img


def visualize_mask(masks, img, path_to_imageseg, crop_row=None, save=False, df=None, cmap_name='fire', alpha_img=0.6, alpha_mask=0.4):
    """
    visualize_mask compatible with standard SEM and Perspective Mode:
      - alpha_img: weight of base SEM image (0.0 for solid RGB masks)
      - alpha_mask: weight/opacity of colorized overlay
    """
    if img is None:
        return None

    if crop_row is not None and crop_row > 0:
        img_vis = img[0:crop_row, :].copy()
    else:
        img_vis = img.copy()
    
    h, w = img_vis.shape[:2]

    if masks is not None and len(masks) > 0:
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        
        masks_arr = np.array([m for m in masks if m is not None])
        if masks_arr.ndim == 2:
            masks_arr = masks_arr[None, ...]
        masks = (masks_arr > 0).astype(np.uint8)
    else:
        if save:
            cv2.imwrite(path_to_imageseg, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        return img_vis

    metric_col, unit_str = detect_metric_column(df)

    if metric_col is not None and not df.empty:
        GLOBAL_MIN = max(0.0, df[metric_col].min() - 1.0)
        GLOBAL_MAX = df[metric_col].max() + 1.0
    else:
        GLOBAL_MIN = 0.0
        GLOBAL_MAX = 60.0

    overlay = np.zeros_like(img_vis, dtype=np.uint8)
    
    is_random = (cmap_name == "random_vivid")
    cmap = None
    if not is_random:
        try:
            if hasattr(cc.cm, cmap_name):
                cmap = getattr(cc.cm, cmap_name)
            elif hasattr(cmr, cmap_name):
                cmap = getattr(cmr, cmap_name)
            else:
                cmap = cm.get_cmap(cmap_name)
        except Exception:
            cmap = cc.cm.fire
    
    for i in range(masks.shape[0]):
        mask_raw = masks[i]
        if mask_raw.shape[0] != h or mask_raw.shape[1] != w:
            mask = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask_raw

        if is_random:
            hue = int(360 * (i / masks.shape[0]))
            hsv_pixel = np.uint8([[[hue // 2, 235, 245]]])
            color_rgb = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]
        else:
            if metric_col is not None and not df.empty:
                particle_id = i + 1
                match = df.loc[df['Particle_ID'] == particle_id, metric_col]
                if not match.empty:
                    val = match.values[0]
                    norm_val = np.clip((val - GLOBAL_MIN) / max(1e-5, (GLOBAL_MAX - GLOBAL_MIN)), 0.0, 1.0)
                else:
                    norm_val = 0.0 
            else:
                norm_val = 0.0 

            rgba_color = cmap(norm_val) 
            color_rgb = (np.array(rgba_color[:3]) * 255).astype(np.uint8)
            
        overlay[mask > 0] = color_rgb

    # Apply alpha weights
    final_seg = cv2.addWeighted(img_vis, alpha_img, overlay, alpha_mask, 0)

    if is_random:
        final_output = final_seg
    else:
        cb_h, cb_pad = 50, 90  
        
        gradient = np.linspace(0, 255, w - (2 * cb_pad)).astype(np.uint8)
        gradient_strip = np.tile(gradient, (cb_h, 1))
        
        colorbar_rgb_raw = cmap(gradient_strip / 255.0) 
        colorbar_rgb = (colorbar_rgb_raw[:, :, :3] * 255).astype(np.uint8)

        # Pre-blend legend bar with identical alpha weights
        bg_mean = np.mean(img_vis, axis=(0, 1)).astype(np.uint8)
        bg_patch = np.full_like(colorbar_rgb, bg_mean)
        colorbar_blended = cv2.addWeighted(bg_patch, alpha_img, colorbar_rgb, alpha_mask, 0)
        cv2.rectangle(colorbar_blended, (0, 0), (colorbar_blended.shape[1]-1, colorbar_blended.shape[0]-1), (255, 255, 255), 2)

        colorbar_y_start = 20
        PIXEL_FONT_SIZE = max(20, int(w * 0.045))
        text_y_pos_ticks = colorbar_y_start + cb_h + 10
        text_y_pos_title = text_y_pos_ticks + PIXEL_FONT_SIZE + 10

        bottom_padding = 42
        legend_h = text_y_pos_title + PIXEL_FONT_SIZE + bottom_padding
        legend_canvas = np.zeros((legend_h, w, 3), dtype=np.uint8)

        legend_canvas[colorbar_y_start:colorbar_y_start+cb_h, cb_pad:w-cb_pad] = colorbar_blended

        # Line 1: Scale endpoints directly aligned with colorbar edges
        legend_canvas = draw_text_scientific(
            legend_canvas, f"{GLOBAL_MIN:.0f} {unit_str}".strip(), 
            (cb_pad, text_y_pos_ticks), PIXEL_FONT_SIZE, anchor="left"
        )
        legend_canvas = draw_text_scientific(
            legend_canvas, f"{GLOBAL_MAX:.0f} {unit_str}".strip(), 
            (cb_pad, text_y_pos_ticks), PIXEL_FONT_SIZE, anchor="right"
        )
        
        # Line 2: Centered statistical summary title
        if metric_col is not None and not df.empty:
            avg_val = df[metric_col].mean()
            std_val = df[metric_col].std()
            clean_col_name = metric_col.replace('_', ' ').replace('deg', '').replace('nm', '').replace('um', '').strip()
            # Round mean/std to whole numbers for display
            label_title = f"Avg {clean_col_name}: {avg_val:.0f} \u00b1 {std_val:.0f} {unit_str}".strip()
        else:
            label_title = "Particle Characterization"
       
        legend_canvas = draw_text_scientific(
            legend_canvas, label_title, 
            (0, text_y_pos_title), PIXEL_FONT_SIZE, anchor="center"
        )
                    
        final_output = np.vstack((final_seg, legend_canvas))

    if save:
        cv2.imwrite(path_to_imageseg, cv2.cvtColor(final_output.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    return final_output
