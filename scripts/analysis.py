# -*- coding: utf-8 -*-
"""
Particle-level metrology and statistical analysis of detected nanoparticles.

Author: Karishma Begum
Year: 2026
"""

from pathlib import Path
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import norm
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sam_visualize import visualize_mask, detect_metric_column

def calculate_nnd(df, ratio):
    if len(df) < 2: return [0.0] * len(df)
    coords = df[['X', 'Y']].values
    dist_matrix = distance.cdist(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)
    return np.min(dist_matrix, axis=1) * ratio

def format_results_for_export(df, ratio=1.0, unit="nm"):
    """
    Builds a clean, export-ready copy of the particle results table: converts
    pixel-based X/Y to physical units, rounds numeric columns, and labels
    each column header with its unit.

    Parameters
    ----------
    df : pd.DataFrame
        The particle results table (as produced by particle_analysis).
    ratio : float
        The image's calibration ratio (physical units per pixel).
    unit : str
        The physical unit label for this image (e.g. "nm" or "um").

    Returns
    -------
    pd.DataFrame
        A new dataframe, safe to pass straight to .to_csv() / to_excel().
    """
    if df is None or df.empty:
        return df

    export_df = df.copy()
    unit_label = unit if unit else "nm"

    # Convert raw-pixel X/Y to the same physical unit as Area
    if 'X' in export_df.columns and 'Y' in export_df.columns:
        export_df['X'] = export_df['X'] * ratio
        export_df['Y'] = export_df['Y'] * ratio

    # Round numeric columns to 2 decimals (Particle_ID stays an int)
    for col in export_df.columns:
        if col == "Particle_ID":
            export_df[col] = export_df[col].astype(int)
        elif pd.api.types.is_numeric_dtype(export_df[col]):
            export_df[col] = export_df[col].round(2)

    # Rename columns to show units -- only columns that exist get renamed
    rename_map = {
        "Particle_ID": "Particle_ID",
        "X": f"X ({unit_label})",
        "Y": f"Y ({unit_label})",
        "Area": f"Area ({unit_label}^2)",
        "Eq_Diameter": f"Eq_Diameter ({unit_label})",
        "Aspect_Ratio": "Aspect_Ratio (ratio)",
        "Circularity": "Circularity (index)",
        "Nearest_Neighbor_Dist": f"Nearest_Neighbor_Dist ({unit_label})",
        # Perspective-mode columns, renamed only if present
        "Contact_Angle_deg": "Contact_Angle (deg)",
        "Height_nm": "Height (nm)",
        "Diameter_nm": "Diameter (nm)",
        "Height_um": "Height (um)",
        "Diameter_um": "Diameter (um)",
        "diameter_px": "Diameter (px)",
        "height_px": "Height (px)",
    }
    existing_renames = {k: v for k, v in rename_map.items() if k in export_df.columns}
    export_df = export_df.rename(columns=existing_renames)

    return export_df

def calculate_surface_coverage(masks_np, raw_img, crop_row=None):
    """
    Calculates surface coverage as the union of the SAM particle masks and
    an Otsu-thresholded pass on the raw image, so coverage reflects both the
    validated particle set and any additional bright material SAM excluded.
    """
    if raw_img is None:
        return 0.0

    img_source = raw_img[:crop_row, :] if (crop_row is not None and 0 < crop_row < raw_img.shape[0]) else raw_img
    gray = cv2.cvtColor(img_source, cv2.COLOR_RGB2GRAY) if img_source.ndim == 3 else img_source
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mask = (otsu_mask > 0).astype(np.uint8)

    if masks_np is not None and len(masks_np) > 0:
        particle_mask = np.clip(np.sum(masks_np, axis=0), 0, 1).astype(np.uint8)
        if crop_row is not None and 0 < crop_row < particle_mask.shape[0]:
            particle_mask = particle_mask[:crop_row, :]
        if particle_mask.shape != otsu_mask.shape:
            particle_mask = particle_mask[:otsu_mask.shape[0], :otsu_mask.shape[1]]
    else:
        particle_mask = np.zeros_like(otsu_mask)

    combined_mask = np.clip(particle_mask + otsu_mask, 0, 1)
    total_pixels = combined_mask.size
    foreground_pixels = np.count_nonzero(combined_mask)
    return float((foreground_pixels / total_pixels) * 100) if total_pixels > 0 else 0.0

def calculate_fft_spacing(img_gray, ratio, save_path=None, save_profile_path=None):
    if img_gray is None: return 0.0, 0
    h, w = img_gray.shape
    dim = min(h, w)

    power_of_2 = 2 ** int(np.floor(np.log2(dim)))
    y0 = (h - power_of_2) // 2
    x0 = (w - power_of_2) // 2
    crop = img_gray[y0:y0 + power_of_2, x0:x0 + power_of_2]

    if save_path is not None:
        cv2.imwrite(str(save_path), crop)

    window = np.outer(np.hanning(power_of_2), np.hanning(power_of_2))
    crop_windowed = crop.astype(np.float64) * window

    dft = np.fft.fft2(crop_windowed)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(np.abs(dft_shift) + 1)

    cy, cx = power_of_2 // 2, power_of_2 // 2

    y, x = np.indices(magnitude_spectrum.shape)
    r_matrix = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    max_radius = power_of_2 // 2
    radial_profile = np.zeros(max_radius)

    for r in range(max_radius):
        radial_profile[r] = np.mean(magnitude_spectrum[r_matrix == r])

    smooth_window = 5
    kernel = np.ones(smooth_window) / smooth_window
    smoothed_profile = np.convolve(radial_profile, kernel, mode="same")

    if save_profile_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(radial_profile, alpha=0.4, label="raw")
        ax.plot(smoothed_profile, label="smoothed")
        ax.axvline(10, color="gray", linestyle="--", linewidth=0.8, label="scan floor (r=10)")
        ax.set_xlabel("Radius (frequency bin)")
        ax.set_ylabel("Mean log magnitude")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(save_profile_path), dpi=120)
        plt.close(fig)

    best_R = 0
    highest_peak_intensity = -1

    for r in range(10, max_radius - 1):
        if smoothed_profile[r] > smoothed_profile[r-1] and smoothed_profile[r] > smoothed_profile[r+1]:
            if smoothed_profile[r] > highest_peak_intensity:
                highest_peak_intensity = smoothed_profile[r]
                best_R = r

    if best_R == 0:
        magnitude_spectrum[cy-18:cy+18, cx-18:cx+18] = 0
        y_peak, x_peak = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        best_R = np.sqrt((x_peak - cx)**2 + (y_peak - cy)**2)
    else:
        centroid_half_width = 8
        r_lo = max(10, best_R - centroid_half_width)
        r_hi = min(max_radius - 1, best_R + centroid_half_width)
        window_radii = np.arange(r_lo, r_hi + 1)
        window_vals = smoothed_profile[r_lo:r_hi + 1]
        weights = np.clip(window_vals - window_vals.min(), 0, None)
        if weights.sum() > 0:
            best_R = float(np.sum(window_radii * weights) / np.sum(weights))

    spacing = (power_of_2 * ratio) / best_R if best_R > 0 else 0
    return spacing, power_of_2

def particle_analysis(masks, boxes, ratio, path_to_csv_file, save=True, unit="nm", 
                      raw_img=None, crop_row=None, user_bins=None, user_x_limit=None, 
                      user_y_limit=None, df_input=None):
    
    actual_ratio = ratio * 1000 if unit.lower() == "um" else ratio
    figs = {}

    # --- STEP 1: DATA ACQUISITION ---
    if df_input is not None:
        df = df_input
    else:
        if masks is None or len(boxes) == 0: return pd.DataFrame(), {}
        try:
            masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        except:
            masks_np = np.asarray(masks)
        
        if masks_np.ndim == 2: masks_np = masks_np[None, ...]
        masks_np = (masks_np > 0).astype(np.uint8)

        rows = []
        for i in range(masks_np.shape[0]):
            mask = masks_np[i]
            if mask.sum() == 0: continue
            lab = label(mask)
            regions = regionprops(lab)
            if not regions: continue
            prop = max(regions, key=lambda r: r.area)
            
            area = float(prop.area) * (actual_ratio**2)
            perim = float(prop.perimeter) * actual_ratio
            rows.append({
                "Particle_ID": i + 1,
                "X": prop.centroid[1],
                "Y": prop.centroid[0],
                "Area": area,
                "Eq_Diameter": 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0,
                "Aspect_Ratio": (prop.major_axis_length / prop.minor_axis_length) if prop.minor_axis_length > 0 else 1.0,
                "Circularity": (4.0 * np.pi * area / (perim**2)) if perim > 0 else 0.0
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df['Nearest_Neighbor_Dist'] = calculate_nnd(df, actual_ratio)
            
            # --- SURFACE COVERAGE CALCULATION ---
            surface_cov = calculate_surface_coverage(masks_np, raw_img, crop_row=crop_row)
            df.attrs['surface_coverage'] = surface_cov

            fft_val = 0.0
            fft_dim = 0
            if raw_img is not None:
                if crop_row is not None and 0 < crop_row < raw_img.shape[0]:
                    fft_source_img = raw_img[:crop_row, :]
                else:
                    fft_source_img = raw_img
                gray = cv2.cvtColor(fft_source_img, cv2.COLOR_RGB2GRAY) if fft_source_img.ndim == 3 else fft_source_img

                img_stem = Path(path_to_csv_file).stem.replace("_results", "")
                fft_img_path = Path(path_to_csv_file).parent / f"{img_stem}_fft.png"
                fft_profile_path = Path(path_to_csv_file).parent / f"{img_stem}_fft_profile.png"

                fft_val, fft_dim = calculate_fft_spacing(
                    gray, actual_ratio,
                    save_path=fft_img_path,
                    save_profile_path=fft_profile_path
                )

            df.attrs['fft_spacing'] = fft_val
            df.attrs['fft_dimension'] = fft_dim
            if save:
                img_stem = Path(path_to_csv_file).stem.replace("_results", "")
                seg_path = str(Path(path_to_csv_file).parent / f"{img_stem}_segmented.png")
                visualize_mask(masks_np, raw_img, seg_path, save=True, df=df, crop_row=crop_row)

    if df.empty: return df, {}

    # --- DYNAMIC METRIC & UNIT DETECTION ---
    metric_col, detected_unit = detect_metric_column(df)
    if metric_col is None or metric_col not in df.columns:
        metric_col = 'Eq_Diameter'
        detected_unit = unit

    working_unit = detected_unit if detected_unit else unit
    metric_label = metric_col.replace('_', ' ').replace('deg', '').replace('nm', '').replace('um', '').strip()

    # --- STEP 2: DYNAMIC CALCULATION OF TRUE BOUNDS ---
    data = df[metric_col].dropna()
    
    dynamic_x_max = float(np.ceil(data.max())) if len(data) > 0 else 100.0
    active_x_limit = user_x_limit if user_x_limit is not None and user_x_limit > 0 else dynamic_x_max
    
    visible_data = data[data <= active_x_limit]
    
    active_bins = user_bins if user_bins is not None and user_bins > 0 else int(np.ceil(np.sqrt(len(visible_data))))
    if active_bins < 5: active_bins = 10
    
    display_bin_width = active_x_limit / active_bins

    counts, edges = np.histogram(visible_data, bins=active_bins, range=(0, active_x_limit))
    dynamic_y_max = int(counts.max()) if len(counts) > 0 else 10
    active_y_limit = user_y_limit if user_y_limit is not None and user_y_limit > 0 else dynamic_y_max

    # --- 3. TRENDS DASHBOARD ---
    fig1 = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Ellipticity Trend", "Local Spacing (NND)", "Surface Coverage"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "domain"}]]
    )
    fig1.add_trace(go.Scatter(x=df[metric_col], y=df['Aspect_Ratio'], mode='markers', marker=dict(color='#e67e22'), text=df['Particle_ID']), row=1, col=1)
    
    nnd_vals = df['Nearest_Neighbor_Dist'] if 'Nearest_Neighbor_Dist' in df.columns else np.zeros(len(df))
    fig1.add_trace(go.Scatter(x=df[metric_col], y=nnd_vals, mode='markers', marker=dict(color='#3498db'), text=df['Particle_ID']), row=1, col=2)
    
    # Surface Coverage Indicator / Gauge
    cov_val = df.attrs.get('surface_coverage', 0.0)
    fig1.add_trace(go.Indicator(
        mode="gauge+number",
        value=cov_val,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2ecc71"},
            'steps': [
                {'range': [0, 50], 'color': "#ecf0f1"},
                {'range': [50, 100], 'color': "#bdc3c7"}
            ]
        }
    ), row=1, col=3)

    fig1.update_xaxes(range=[0, active_x_limit], title_text=f"{metric_label} ({working_unit})", row=1, col=1)
    fig1.update_xaxes(range=[0, active_x_limit], title_text=f"{metric_label} ({working_unit})", row=1, col=2)
    fig1.update_yaxes(title_text="Aspect Ratio", row=1, col=1)
    fig1.update_yaxes(title_text=f"Spacing ({working_unit})", row=1, col=2)
    fig1.update_layout(height=450, template="plotly_white", showlegend=False)
    figs['trends'] = fig1

    # --- 4. SIZE DISTRIBUTION & MORPHOLOGY ---
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=(f"{metric_label} Distribution & Gaussian Fit", f"Circularity vs {metric_label}"))
    
    if len(visible_data) > 1:
        mu, std = norm.fit(visible_data)
        x_fit = np.linspace(0, active_x_limit, 300)
        raw_pdf_y = norm.pdf(x_fit, mu, std)
        y_fit_scaled = raw_pdf_y * len(visible_data) * display_bin_width
    else:
        mu, std = (data.mean(), data.std()) if len(data) > 0 else (0.0, 0.0)
        x_fit, y_fit_scaled = np.array([0, active_x_limit]), np.array([0, 0])

    fig2.add_trace(go.Histogram(
        x=visible_data, marker_color='#2ecc71', opacity=0.6, name="Histogram",
        xbins=dict(start=0, end=active_x_limit, size=display_bin_width),
        autobinx=False, showlegend=False
    ), row=1, col=1)
    
    fig2.add_trace(go.Scatter(
        x=x_fit, y=y_fit_scaled, mode='lines', 
        line=dict(color='red', width=3), 
        name='Gaussian Fit'
    ), row=1, col=1)
    
    circ_vals = df['Circularity'] if 'Circularity' in df.columns else np.ones(len(df))
    fig2.add_trace(go.Scatter(
        x=df[metric_col], y=circ_vals, 
        mode='markers', marker=dict(color='#9b59b6'),
        name='Circularity Trace'
    ), row=1, col=2)
    
    fig2.add_annotation(
        text=f"<b>Stats</b><br>μ: {mu:.1f} {working_unit}<br>σ: {std:.1f} {working_unit}",
        xref="x2", yref="y2",
        x=active_x_limit * 0.95, y=0.95,
        showarrow=False, align="right",
        bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1
    )
    
    x_tick_spacing = active_x_limit / 10
    fig2.update_xaxes(range=[0, active_x_limit], dtick=x_tick_spacing, title_text=f"{metric_label} ({working_unit})", row=1, col=1)
    fig2.update_yaxes(range=[0, active_y_limit * 1.15], title_text="Frequency", row=1, col=1)
    
    fig2.update_xaxes(range=[0, active_x_limit], dtick=x_tick_spacing, title_text=f"{metric_label} ({working_unit})", row=1, col=2)
    fig2.update_yaxes(range=[0, 1.5], title_text="Circularity Index", row=1, col=2)
    
    fig2.update_layout(
        height=450, 
        template="plotly_white", 
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    figs['dist'] = fig2

    if save and df_input is None: 
        df.to_csv(path_to_csv_file, index=False)
        
    return df, figs

# =====================================================================
# Static matplotlib export of a 4-panel publication figure, built from
# the same particle-level dataframe produced by particle_analysis().
# =====================================================================
def generate_publication_figure(df, save_path=None, metric_col="Eq_Diameter",
                                 metric_label="Equivalent Diameter", unit="nm",
                                 save=True, cmap_name="viridis"):
    """
    Generates a 4-panel publication-style figure from a particle-level
    dataframe: (a) size distribution with Gaussian fit, (b) aspect ratio
    vs. size, (c) nearest-neighbor distance vs. size, (d) circularity vs.
    size as a hexbin density plot.

    Args:
        df: particle-level dataframe; must contain metric_col,
            'Aspect_Ratio', 'Circularity', 'Nearest_Neighbor_Dist'.
        save_path: PNG path to write the figure to, if provided.
        save: if True, closes the figure after saving and returns
            save_path; if False, returns the open Figure object for
            display via st.pyplot(fig).
        cmap_name: colormap theme name, resolved via colorcet -> cmasher
            -> matplotlib fallback (same priority as sam_visualize.py).
            "random_vivid" has no continuous equivalent and falls back
            to "viridis".

    Returns:
        The saved path (save=True) or the matplotlib Figure (save=False).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import matplotlib.cm as mpl_cm
    from matplotlib.colors import Normalize
    from scipy.stats import norm as scipy_norm

    if df is None or df.empty or metric_col not in df.columns:
        return None

    # Resolve the theme name into a matplotlib-compatible colormap
    def _resolve_cmap(name):
        if name == "random_vivid":
            return plt.colormaps["viridis"]
        try:
            import colorcet as cc
            if hasattr(cc.cm, name):
                return getattr(cc.cm, name)
        except Exception:
            pass
        try:
            import cmasher as cmr
            if hasattr(cmr, name):
                return getattr(cmr, name)
        except Exception:
            pass
        try:
            return plt.colormaps[name]
        except Exception:
            return plt.colormaps["viridis"]

    theme_cmap = _resolve_cmap(cmap_name)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 13,
    })
    TITLE_COLOR = "#1a1a1a"
    GRID_COLOR = "#e0e0e0"

    values = df[metric_col].dropna().values
    if len(values) < 2:
        return None

    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28)

    cmap = theme_cmap
    norm_metric = Normalize(vmin=values.min(), vmax=values.max())

    # --- (a) Size distribution with Gaussian fit ---
    ax_a = fig.add_subplot(gs[0, 0])
    mu, std = scipy_norm.fit(values)
    n_bins = min(15, max(10, int(np.sqrt(len(values)) / 1.5)))
    counts, bin_edges, patches = ax_a.hist(
        values, bins=n_bins, color="#8fd9c4", edgecolor="white", alpha=0.85
    )
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        patch.set_facecolor(cmap(norm_metric(left_edge)))
    x_fit = np.linspace(0, values.max() * 1.15, 300)
    bin_width = bin_edges[1] - bin_edges[0]
    y_fit = scipy_norm.pdf(x_fit, mu, std) * len(values) * bin_width
    ax_a.plot(x_fit, y_fit, color="#d62728", linewidth=2.2,
              label=f"Gaussian Curve\n\u03bc={mu:.1f} {unit}, \u03c3={std:.1f} {unit}")
    ax_a.set_xlabel(f"{metric_label} ({unit})")
    ax_a.set_ylabel("Frequency")
    ax_a.set_title("(a) Size Distribution", loc="left", fontweight="bold", color=TITLE_COLOR)
    ax_a.legend(frameon=False, fontsize=9, loc="upper right")
    ax_a.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax_a.spines[["top", "right"]].set_visible(False)

    # --- (b) Aspect ratio vs size, colored by circularity ---
    ax_b = fig.add_subplot(gs[0, 1])
    if "Aspect_Ratio" in df.columns and "Circularity" in df.columns:
        sc_b = ax_b.scatter(df[metric_col], df["Aspect_Ratio"], c=df["Circularity"],
                             cmap=theme_cmap, s=22, alpha=0.75, edgecolors="none")
        cb_b = fig.colorbar(sc_b, ax=ax_b, pad=0.02)
        cb_b.set_label("Circularity", fontsize=9)
    ax_b.set_xlabel(f"{metric_label} ({unit})")
    ax_b.set_ylabel("Aspect Ratio")
    ax_b.set_title("(b) Shape vs. Particle Size", loc="left", fontweight="bold", color=TITLE_COLOR)
    ax_b.grid(color=GRID_COLOR, linewidth=0.6)
    ax_b.spines[["top", "right"]].set_visible(False)

    # --- (c) NND vs size, colored by size ---
    ax_c = fig.add_subplot(gs[1, 0])
    if "Nearest_Neighbor_Dist" in df.columns:
        sc_c = ax_c.scatter(df[metric_col], df["Nearest_Neighbor_Dist"], c=values,
                             cmap=theme_cmap, s=22, alpha=0.75, edgecolors="none")
        cb_c = fig.colorbar(sc_c, ax=ax_c, pad=0.02)
        cb_c.set_label(f"{metric_label} ({unit})", fontsize=9)
    ax_c.set_xlabel(f"{metric_label} ({unit})")
    ax_c.set_ylabel(f"Nearest-Neighbor Distance ({unit})")
    ax_c.set_title("(c) Local Spacing vs. Particle Size", loc="left", fontweight="bold", color=TITLE_COLOR)
    ax_c.grid(color=GRID_COLOR, linewidth=0.6)
    ax_c.spines[["top", "right"]].set_visible(False)

    # --- (d) Circularity density (hexbin) ---
    ax_d = fig.add_subplot(gs[1, 1])
    if "Circularity" in df.columns:
        hb = ax_d.hexbin(df[metric_col], df["Circularity"], gridsize=28,
                          cmap=theme_cmap, mincnt=1)
        cb_d = fig.colorbar(hb, ax=ax_d, pad=0.02)
        cb_d.set_label("Particle count", fontsize=9)
    ax_d.set_xlabel(f"{metric_label} ({unit})")
    ax_d.set_ylabel("Circularity Index")
    ax_d.set_title("(d) Circularity Density", loc="left", fontweight="bold", color=TITLE_COLOR)
    ax_d.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Particle-Level Morphological and Spatial Analysis",
                  fontsize=13, fontweight="bold", y=0.98)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")

    if save:
        plt.close(fig)
        return save_path
    else:
        # Returned open so it can be displayed live via st.pyplot(fig)
        return fig