# -*- coding: utf-8 -*-
"""
Streamlit front-end for the SEM nanoparticle detection, segmentation, and metrology pipeline.

Author: Karishma Begum
Year: 2026
"""

import streamlit as st
import sys
import os
import shutil
import time
from PIL import Image
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2
import warnings
import logging
from fpdf import FPDF
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
OUT_BASE_DIR = PROJECT_ROOT / "outputs"
OUT_BASE_DIR.mkdir(exist_ok=True)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# --- Import Validator ---
#try:
    #from validator import check_image_viability
    #validator_loaded = True
#except ImportError:
    #validator_loaded = False

# --- Scale Bar Metric Detection ---
TEMPLATE_DIR = SCRIPTS_DIR / "sem_templates"
try:
    from metrology import SEMCalibrator
    calibrator = SEMCalibrator(template_dir=str(TEMPLATE_DIR))
    metrology_loaded = True
except ImportError:
    metrology_loaded = False
except Exception as e:
    st.sidebar.error(f"SEMCalibrator failed to initialize: {e}")
    metrology_loaded = False

# -------------------
# Page layout setting
# --------------------
st.set_page_config(layout="wide", page_title="SEM Analyzer")

st.markdown(
    """
    <style>
      .block-container {
          padding-top: 6rem;   
          padding-bottom: 1rem;
          max-width: 100%;
      }
      .stMetric {
          background-color: #ffffff;
          padding: 15px;
          border-radius: 8px;
          border: 1px solid #eee;
      }
      [data-testid="stAlert"] p {
          font-size: 15px !important;
          font-weight: 600 !important;
      }
      /* for the labels on the UI - the total partciles etc */
      [data-testid="stMetricLabel"],
      [data-testid="stMetricLabel"] p {
          font-size: 25px !important;
          color: #333333 !important;
          font-weight: 700 !important;
      }
      /* for the values on the UI- for the avg psrtcile size etc */
      [data-testid="stMetricValue"] {
          font-size: 28px !important;
          color: #0d1b2a !important;
          font-weight: 600 !important;
      }
      /* Small Green Run Button */
      div.stButton > button:first-child {
          background-color: #28a745;
          color: white;
          padding: 0.2rem 1rem;
          border-radius: 5px;
          border: none;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Silence warnings/log noise
# -----------------------------
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("torch.classes").setLevel(logging.ERROR)

# -----------------------------
# Session State & Reset Logic
# -----------------------------
if "processed_data" not in st.session_state:
    st.session_state.processed_data = []
if "report_selection" not in st.session_state:
    st.session_state.report_selection = []
if "roi_mode" not in st.session_state:
    st.session_state.roi_mode = "NAVIGATE"
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "roi_json" not in st.session_state:
    st.session_state.roi_json = None
if "skipped_log" not in st.session_state:
    st.session_state.skipped_log = []
if "dataset_sorter" not in st.session_state:
    import importlib
    try:
        import clustering
        st.session_state.dataset_sorter = clustering.DatasetSorter()
    except ImportError:
        try:
            from scripts import clustering
            st.session_state.dataset_sorter = clustering.DatasetSorter()
        except ImportError:
            st.session_state.dataset_sorter = None

def clear_workspace():
    """Wipes output folders, session data, and resets ROI tools."""
    for folder in OUT_BASE_DIR.glob('*'):
        if folder.is_dir():
            shutil.rmtree(folder)
    st.session_state.processed_data = []
    st.session_state.report_selection = []
    st.session_state.skipped_log = []
    st.session_state.roi_mode = "NAVIGATE"
    st.session_state.roi_json = None
    if st.session_state.dataset_sorter is not None:
        st.session_state.dataset_sorter.summary_df = None
    st.session_state.canvas_key += 1 
    st.rerun()

# -----------------------------
# Load scripts
# -----------------------------
try:
    from detector import object_detection
    from data_reader import load
    from analysis import particle_analysis
    from sam_visualize import visualize_mask, create_semantic_transformation
    from perspective import perspective_detection, build_perspective_figures
    scripts_loaded = True
except ImportError as e:
    st.sidebar.error(f"Error loading scripts: {e}")
    scripts_loaded = False

# -----------------------------
# ROI Scaling Logic
# -----------------------------
def crop_from_canvas_scaled(img_rgb, json_data, scale):
    if json_data is None: return img_rgb, None
    objs = json_data.get("objects", [])
    if not objs: return img_rgb, None
    obj = objs[-1]
    h, w = img_rgb.shape[:2]
    def unscale(v): return float(v) / float(scale)
    if obj.get("type") == "rect":
        l, t = int(max(0, unscale(obj.get("left", 0)))), int(max(0, unscale(obj.get("top", 0))))
        wd, ht = int(max(1, unscale(obj.get("width", 1)))), int(max(1, unscale(obj.get("height", 1))))
        r, b = int(min(w, l + wd)), int(min(h, t + ht))
        return img_rgb[t:b, l:r].copy(), {"shape": "rect", "x1": l, "y1": t, "x2": r, "y2": b}
    if obj.get("type") == "circle":
        ld, td, rd = float(obj.get("left", 0)), float(obj.get("top", 0)), float(obj.get("radius", 1))
        cx, cy, r = unscale(ld + rd), unscale(td + rd), unscale(rd)
        x1, y1, x2, y2 = int(max(0, cx-r)), int(max(0, cy-r)), int(min(w, cx+r)), int(min(h, cy+r))
        crop = img_rgb[y1:y2, x1:x2].copy()
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.circle(mask, (int(cx-x1), int(cy-y1)), int(r), 255, -1)
        crop[mask == 0] = 0 
        return crop, {"shape": "circle", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "mask": mask}
    return img_rgb, None

# -----------------------------
# Analysis Report - PDF Generation
# -----------------------------
def generate_final_report(selected_items):
    """Generates a PDF containing only handpicked analysis results with tables and StdDev."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Cover Page ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(200, 60, txt="Final Segmented Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"Total Selected Samples: {len(selected_items)}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Report Generated: {time.strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')

    for item in selected_items:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(0, 12, txt=f"Sample: {item['name']}", ln=True, fill=True)
        
        dia_col = 'Eq_Diameter' if 'Eq_Diameter' in item['df'].columns else 'Diameter_nm'
        std_dev = item['df'][dia_col].std() if not item['df'].empty else 0.0
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt="Core Characterization Metrics:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 7, txt=f"- Total Particle Count: {item['count']}", ln=True)
        pdf.cell(0, 7, txt=f"- Average Particle Size: {item['avg_size']:.0f} \u00b1 {std_dev:.0f} {item['unit']}", ln=True)
        
        if 'Contact_Angle_deg' in item['df'].columns:
            avg_ca = item['df']['Contact_Angle_deg'].mean()
            std_ca = item['df']['Contact_Angle_deg'].std()
            pdf.cell(0, 7, txt=f"- Mean Contact Angle: {avg_ca:.1f}\u00b0 \u00b1 {std_ca:.1f}\u00b0", ln=True)
        else:
            pdf.cell(0, 7, txt=f"- Nearest Neighbor Distance (NND): {item['nnd']:.2f} {item['unit']}", ln=True)
            
        pdf.ln(5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            h_path = os.path.join(tmp_dir, "h.png")
            cv2.imwrite(h_path, item['heatmap_img'])
            curr_y = pdf.get_y()
            pdf.image(h_path, x=10, y=curr_y, w=90)
            
            s_path = os.path.join(tmp_dir, "s.png")
            cv2.imwrite(s_path, cv2.cvtColor(item['semantic_img'], cv2.COLOR_RGB2BGR))
            pdf.image(s_path, x=105, y=curr_y, w=90)
            
        pdf.ln(75) 

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="Raw Particle Data (Top 50 Samples)", ln=True)
        pdf.cell(0, 7, txt=f"- Surface Coverage Area: {item['coverage']:.2f}%", ln=True)
        
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(60, 8, "Particle ID", 1, 0, 'C', True)
        pdf.cell(60, 8, f"Diameter ({item['unit']})", 1, 1, 'C', True)
        
        pdf.set_font("Arial", size=9)
        for _, row in item['df'].head(50).iterrows():
            pdf.cell(60, 7, str(int(row['Particle_ID'])), 1, 0, 'C')
            pdf.cell(60, 7, f"{row[dia_col]:.2f}", 1, 1, 'C')

    return pdf.output(dest='S').encode('latin-1')

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("SEM Analysis Tool")
    uploaded_files = st.file_uploader("Upload files", type=["jpg", "png", "tif"], accept_multiple_files=True)

    # Analysis mode selector
    st.header("Analysis Mode")
    analysis_mode = st.selectbox(
        "Select Micrograph Orientation",
        ["Standard Mode", "Perspective Mode"],
        help="Use Perspective Mode to extract contact angles, heights, and base diameters from tilted SEM micrographs."
    )

    st.header("Display Options")
    viz_mode = st.selectbox(
        "Select Visualization Mode",
        ["Standard Transformation", "Semantic Transformation"],
        help="creates a vertical split-field transition figures."
    )
    
    if viz_mode == "Semantic Transformation":
        split_ratio = st.slider("Transformation Line Position", 0.0, 1.0, 0.5)

    if analysis_mode == "Perspective Mode":
        persp_view_mode = st.selectbox(
            "Perspective Overlay Style",
            ["Clean Geometry Overlay", "Contact Angle Heatmap"],
            help=(
                "Clean Geometry shows just the measured baseline/height/angle "
                "lines drawn on the original image -- nothing else touched. "
                "Contact Angle Heatmap additionally fills each particle with a "
                "translucent color keyed to its angle, plus a legend bar."
            )
        )
    else:
        persp_view_mode = "Contact Angle Heatmap"

    st.markdown("### Visualization Settings")
    palette_options = {
        "🔥 Colorcet Fire (Max Luminous)": "fire",
        "🌿 Colorcet Blue-Green-Yellow": "bgyw",
        "✨ CMasher Amber (Golden Glow)": "amber",
        "☄️ CMasher Ember (Blazing Crimson)": "ember",
        "🌌 CMasher Infinity (Neon Cyclic)": "infinity",
        "🌋 Matplotlib Inferno (Vibrant Red-Yellow)": "inferno",
        "⚡ Matplotlib Plasma (Neon Pink-Yellow)": "plasma",
        "🧬 Gnuplot2 (High-Saturated Multi)": "gnuplot2",
        "🌡️ Gist Heat (Thermal Highlight)": "gist_heat",
        "🌈 Random Vivid Colors (Instance Breakout)": "random_vivid"
    }
    selected_palette_name = st.sidebar.selectbox(
        "Select Nanoparticle Color Theme:",
        options=list(palette_options.keys())
    )
    user_cmap = palette_options[selected_palette_name]

    if st.session_state.report_selection:
        st.markdown("---")
        st.subheader("Report Builder")
        st.success(f"{len(st.session_state.report_selection)} items in queue")
        
        final_pdf_data = generate_final_report(st.session_state.report_selection)
        st.download_button(
            "Download Report",
            data=final_pdf_data,
            file_name="Final_Analysis_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        if st.button("Clear Report Queue", use_container_width=True):
            st.session_state.report_selection = []
            st.rerun()

    st.header("ROI Tools")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Navigate", use_container_width=True): st.session_state.roi_mode = "NAVIGATE"
        if st.button("Rect Area", use_container_width=True): st.session_state.roi_mode = "RECT"
    with c2:
        if st.button("Circle Area", use_container_width=True): st.session_state.roi_mode = "CIRCLE"
        if st.button("Clear All", use_container_width=True, type="secondary"): clear_workspace()
    
    st.caption(f"Active Tool: {st.session_state.roi_mode}")

    st.header("Calibration")
    pixel_val = st.number_input("Fallback Pixel [PX]", value=1024)
    actual_size = st.number_input("Fallback Actual Size", value=30.0)
    unit_input = st.selectbox("Unit", ["nm", "um"])
    fallback_ratio = round(actual_size / pixel_val, 6) if pixel_val else 0.0

# -----------------------------
# Main Workspace
# -----------------------------
if not uploaded_files:
    st.info("Import SEM images in the sidebar to begin the analysis.")
else:
    head_col, btn_col = st.columns([8, 2])
    head_col.subheader("SEM Analysis Tool")
    run_btn = btn_col.button("Run Analysis")

    if run_btn:
        st.session_state.processed_data = [] 
        st.session_state.skipped_log = [] # Reset skipped log on run
        monitor = st.empty()
        progress_text = st.empty()
        prog_bar = st.empty()
        manifest_records = []

        for idx, file in enumerate(uploaded_files):
            p_val = (idx + 1) / len(uploaded_files)
            progress_text.markdown(f"**Progress:** Processing {idx+1} of {len(uploaded_files)}...")
            prog_bar.progress(p_val)

            img_stem = Path(file.name).stem
            img_folder = OUT_BASE_DIR / img_stem
            img_folder.mkdir(exist_ok=True)
            tmp_path = os.path.join("/tmp", file.name)
            with open(tmp_path, "wb") as f: f.write(file.getbuffer())

            img_x, img_meta, _ = load(tmp_path)

            # --- Outlier Detection Check (Validator) ---
            #if validator_loaded:
               #is_viable, reason, metrics = check_image_viability(img_x)
               #if not is_viable:
                    #st.session_state.skipped_log.append({
                        #"File": file.name,
                        #"Reason": reason,
                    #})
                    #manifest_records.append({
                        #"file_name": file.name, 
                        #"particle_count": 0, 
                        #"avg_size_nm": 0.0,
                        #"display_label": "Filtered Out", 
                        #"tmp_filepath": tmp_path
                    #})
                    #continue # Skip to next image

            if metrology_loaded:
                dyn_ratio, dyn_unit, audit_img = calibrator.get_calibration(img_x, fallback_ratio)
            else:
                dyn_ratio, dyn_unit, audit_img = fallback_ratio, unit_input, None

            with monitor.container():
                st.markdown(f"#### Metrology Audit: `{file.name}`")
                aud_c1, aud_c2 = st.columns([2, 1])
                with aud_c1:
                    st.image(img_x, width=450)
                with aud_c2:
                    if audit_img is not None:
                        st.image(audit_img, use_column_width=True)
                        st.success(f"Calibration Coefficient: {dyn_ratio:.2f} {dyn_unit}/px")

            img_for_ai = img_x
            if st.session_state.roi_mode in ["RECT", "CIRCLE"] and st.session_state.roi_json:
                h_orig, w_orig = img_x.shape[:2]
                scale = 900 / w_orig
                img_for_ai, _ = crop_from_canvas_scaled(img_x, st.session_state.roi_json, scale)

            # Route to the perspective-mode or standard-mode pipeline
            if analysis_mode == "Perspective Mode":
                (
                    masks, 
                    boxes, 
                    contact_data, 
                    img_used, 
                    crop_row, 
                    label_found, 
                    contact_panel, 
                    avg_spacing_nnd, 
                    global_fft_spacing
                ) = perspective_detection(
                    img_for_ai, 
                    img_meta, 
                    tmp_path, 
                    conf_thresh=0.15,
                    min_area=15, 
                    nm_per_pixel=dyn_ratio
                )

                if contact_data:
                    df_res = pd.DataFrame(contact_data)
                    df_res.insert(0, 'Particle_ID', range(1, len(df_res) + 1))
                    df_res.rename(columns={
                        'contact_angle': 'Contact_Angle_deg',
                        'height_px': f'Height_{dyn_unit}',
                        'diameter_px': f'Diameter_{dyn_unit}'
                    }, inplace=True)
                    
                    df_res[f'Height_{dyn_unit}'] = df_res[f'Height_{dyn_unit}'] * dyn_ratio
                    df_res[f'Diameter_{dyn_unit}'] = df_res[f'Diameter_{dyn_unit}'] * dyn_ratio
                    df_res['Nearest_Neighbor_Dist'] = avg_spacing_nnd
                    df_res['Eq_Diameter'] = df_res[f'Diameter_{dyn_unit}']
                    df_res.attrs['fft_spacing'] = global_fft_spacing
                    df_res.attrs['fft_dimension'] = img_used.shape[0] if img_used is not None else 0
                else:
                    df_res = pd.DataFrame()
                
                figs_res = build_perspective_figures(df_res, unit=dyn_unit)
                
                seg_path = str(img_folder / f"{img_stem}_segmented.png")
                
                if len(masks) > 0 and persp_view_mode == "Heatmap for Contact Angle":
                    final_persp_img = visualize_mask(
                        masks=masks, 
                        img=contact_panel, 
                        path_to_imageseg=seg_path, 
                        crop_row=crop_row, 
                        save=True, 
                        df=df_res, 
                        cmap_name=user_cmap, 
                        alpha_img=0.6, 
                        alpha_mask=0.4
                    )
                else:
                    cv2.imwrite(seg_path, cv2.cvtColor(contact_panel, cv2.COLOR_RGB2BGR))

            else:
                # Standard Mode Pipeline Execution
                masks, boxes, img_used, crop_row, label_found = object_detection(img_for_ai, img_meta, tmp_path)

                csv_path = str(img_folder / f"{img_stem}_results.csv")
                df_res, figs_res = particle_analysis(
                    masks, boxes, dyn_ratio, csv_path, 
                    save=True, unit=dyn_unit, raw_img=img_for_ai, 
                    crop_row=crop_row
                )
                
                seg_path = str(img_folder / f"{img_stem}_segmented.png")
                visualize_mask(
                    masks, img_for_ai, seg_path, crop_row=crop_row, 
                    save=True, df=df_res, cmap_name=user_cmap
                )
            
            # Save Legend-Free Version for Publication
            try:
                full_segmented_mat = cv2.imread(seg_path)
                if full_segmented_mat is not None:
                    if crop_row is not None and crop_row > 0:
                        paper_clean_mat = full_segmented_mat[0:crop_row, :]
                    else:
                        h_mask = full_segmented_mat.shape[0]
                        paper_clean_mat = full_segmented_mat[0:int(h_mask * 0.88), :]
                    
                    paper_out_path = str(img_folder / f"{img_stem}_segmented_image.png")
                    cv2.imwrite(paper_out_path, paper_clean_mat)
            except Exception as e:
                st.sidebar.warning(f"Failed to generate paper-ready crop layout: {e}")
            
            # Store results in session state
            dia_col = 'Eq_Diameter' if 'Eq_Diameter' in df_res.columns else 'Diameter_nm'
            avg_sz = df_res[dia_col].mean() if not df_res.empty else 0.0
            std_sz = df_res[dia_col].std() if not df_res.empty else 0.0
            avg_sz_nm = avg_sz if dyn_unit == "nm" else avg_sz * 1000

            st.session_state.processed_data.append({
                "name": file.name,
                "count": len(masks),
                "seg": seg_path,
                "df": df_res,
                "figs": figs_res, 
                "folder": str(img_folder),
                "ratio": dyn_ratio, 
                "unit": dyn_unit,
                "masks": masks,
                "crop_row": crop_row,
                "original_img": img_for_ai,
                "mode": analysis_mode
            })
            
            manifest_records.append({
                "file_name": file.name,
                "particle_count": len(masks),
                "avg_size_nm": avg_sz_nm,
                "display_label": f"{avg_sz:.1f} ± {std_sz:.1f} {dyn_unit}" if len(masks) > 0 else "0.00",
                "tmp_filepath": tmp_path
            })

        if st.session_state.dataset_sorter is not None:
            st.session_state.dataset_sorter.summary_df = pd.DataFrame(manifest_records)
        monitor.empty()
        progress_text.success(f"Analysis Completed!")

    # --- Outlier Report Display ---
    #if st.session_state.skipped_log:
        #with st.expander("Quality Control: Auto-Excluded Images", expanded=True):
            #total_count = len(uploaded_files)
            #failed_count = len(st.session_state.skipped_log)
            #passed_count = total_count - failed_count
 
            #st.markdown(
               # f"**Total uploaded: {total_count}**  |  **Passed: {passed_count}**  |  **Excluded: {failed_count}**"
            #)
            
            #st.warning("The following micrographs did not satisfy the shape-based quality criteria required for reliable quantitative analysis and were excluded prior to detection.")
            #st.table(st.session_state.skipped_log)

    # 2. RESULT FEED
    if st.session_state.processed_data:
        st.divider()
        st.subheader("Completed Analysis Stream")
        
        for res in reversed(st.session_state.processed_data):
            cur_unit = res['unit']
            surf_cov = res['df'].attrs.get('surface_coverage', 0.0)
            is_persp = res.get("mode") == "Perspective Mode"
            dia_key = 'Eq_Diameter' if 'Eq_Diameter' in res['df'].columns else f'Diameter_{cur_unit}'
            
            with st.container():
                st.markdown(f"### {res['name']} `{['Standard Top-Down Mode', 'Perspective Contact Angle'][is_persp]}`")
                c_vis, c_info = st.columns([7, 3])
                
                with c_vis:
                    paper_img = create_semantic_transformation(
                        res['original_img'], 
                        res['masks'], 
                        split_ratio=split_ratio if 'split_ratio' in locals() else 0.5, 
                        crop_row=res['crop_row'],
                        df=res['df'],
                        cmap_name=user_cmap
                    )

                    if viz_mode == "Semantic Transformation":
                        st.image(paper_img, use_column_width=True)
                        img_btn_data = cv2.imencode('.png', cv2.cvtColor(paper_img, cv2.COLOR_RGB2BGR))[1].tobytes()
                        img_file_name = f"semantic_image_{res['name']}.png"
                    else:
                        st.image(res['seg'], use_column_width=True)
                        with open(res['seg'], "rb") as file:
                            img_btn_data = file.read()
                        img_file_name = f"{Path(res['name']).stem}.png"

                    if st.button(f"Add {res['name']} to Report", key=f"add_rpt_{res['name']}", use_container_width=True):
                        selection = {
                            "name": res['name'],
                            "count": res['count'],
                            "coverage": surf_cov,
                            "avg_size": res['df'][dia_key].mean() if not res['df'].empty else 0.0,
                            "nnd": res['df']['Nearest_Neighbor_Dist'].mean() if not res['df'].empty else 0.0,
                            "unit": res['unit'],
                            "heatmap_img": cv2.imread(res['seg']),
                            "semantic_img": paper_img,
                            "df": res['df']
                        }
                        st.session_state.report_selection.append(selection)
                        st.toast(f"Added {res['name']} to queue!")

                    st.markdown("---")
                    col_img, col_csv = st.columns(2)

                    with col_img:
                        st.download_button(
                            label="Download Image",
                            data=img_btn_data,
                            file_name=img_file_name,
                            mime="image/png",
                            key=f"dl_img_{res['name']}"
                        )

                    with col_csv:
                        from analysis import format_results_for_export
                        export_df = format_results_for_export(
                            res['df'], ratio=res['ratio'], unit=res['unit']
                        )
                        csv_data = export_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV Results",
                            data=csv_data,
                            file_name=f"{Path(res['name']).stem}_results.csv",
                            mime="text/csv",
                            key=f"dl_csv_{res['name']}"
                        )
                
                with c_info:
                    st.metric("Total Particles", f"{res['count']}")
                    if not res['df'].empty:
                        avg_size = res['df'][dia_key].mean()
                        st.metric("Avg Particle Size", f"{avg_size:.1f} {cur_unit}")
                        
                        if is_persp and 'Contact_Angle_deg' in res['df'].columns:
                            avg_ca = res['df']['Contact_Angle_deg'].mean()
                            std_ca = res['df']['Contact_Angle_deg'].std()
                            st.metric("Mean Contact Angle (\u03b8)", f"{avg_ca:.1f}\u00b0 \u00b1 {std_ca:.1f}\u00b0")
                            
                            if f'Height_{cur_unit}' in res['df'].columns:
                                avg_h = res['df'][f'Height_{cur_unit}'].mean()
                                st.metric("Avg Particle Height", f"{avg_h:.1f} {cur_unit}")
                    
                    st.markdown("---")
                    avg_nnd = res['df']['Nearest_Neighbor_Dist'].mean() if not res['df'].empty else 0.0
                    st.metric(f"Avg Spacing (NND)", f"{avg_nnd:.1f} {cur_unit}")
                    
                    fft_val = res['df'].attrs.get('fft_spacing', 0.0)
                    fft_dim = res['df'].attrs.get('fft_dimension', 0)
                    
                    st.metric(
                        label="Global FFT Spacing", 
                        value=f"{fft_val:.1f} {cur_unit}",
                        delta=f"Matrix: {fft_dim}x{fft_dim} px", 
                        delta_color="off"
                    )
                    cov_val = res['df'].attrs.get('surface_coverage', 0.0)
                    st.metric("Surface Coverage", f"{cov_val:.1f}%")

                with st.expander("ADVANCED CHARACTERIZATION DASHBOARDS", expanded=True):
                    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Morphology Trends", "Size Distribution Trends", "Advanced Analysis"])
                    
                    with tab1:
                        from analysis import format_results_for_export
                        from sam_visualize import highlight_particle
                        display_df = format_results_for_export(
                            res['df'], ratio=res['ratio'], unit=res['unit']
                        )

                        st.caption("Click a row to highlight that exact particle on the segmented image below.")

                        selected_particle_id = None
                        try:
                            # Modern Streamlit: row-click selection on the dataframe itself.
                            event = st.dataframe(
                                display_df,
                                use_container_width=True,
                                on_select="rerun",
                                selection_mode="single-row",
                                key=f"raw_data_select_{res['name']}"
                            )
                            sel_rows = getattr(getattr(event, "selection", None), "rows", [])
                            if sel_rows:
                                selected_particle_id = int(display_df.iloc[sel_rows[0]]["Particle_ID"])
                        except TypeError:
                            # Fallback picker for Streamlit versions without row-selection support
                            st.dataframe(display_df, use_container_width=True)
                            particle_ids = display_df["Particle_ID"].tolist() if "Particle_ID" in display_df.columns else []
                            if particle_ids:
                                picked = st.selectbox(
                                    "Highlight Particle_ID",
                                    options=["None"] + [str(p) for p in particle_ids],
                                    key=f"raw_data_picker_{res['name']}"
                                )
                                selected_particle_id = None if picked == "None" else int(picked)

                        if selected_particle_id is not None:
                            hl_img, hl_crop = highlight_particle(
                                res['masks'], res['original_img'], selected_particle_id,
                                crop_row=res['crop_row']
                            )
                            if hl_img is not None:
                                st.success(f"Highlighting Particle_ID {selected_particle_id}")
                                hc1, hc2 = st.columns([3, 2])
                                with hc1:
                                    st.image(hl_img, caption=f"Full image — Particle {selected_particle_id} outlined", use_column_width=True)
                                with hc2:
                                    if hl_crop is not None:
                                        st.image(hl_crop, caption="Zoomed-in crop", use_column_width=True)
                            else:
                                st.warning(f"Could not locate Particle_ID {selected_particle_id} on the image.")
                    
                    with tab2:
                        if "trends" in res['figs']:
                            st.plotly_chart(res['figs']["trends"], use_container_width=True)

                    with tab3:
                        st.markdown("##### Axis Tuning")
                        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
                        with ctrl_col1:
                            bin_val = st.slider("Histogram Bins", 0, 100, 0, key=f"bins_{res['name']}")
                        with ctrl_col2:
                            x_lim = st.number_input("Max Diameter (nm)", 0, 1000, 0, key=f"xlim_{res['name']}")
                        with ctrl_col3:
                            y_lim = st.number_input("Max Freq (Count)", 0, 1000, 0, key=f"ylim_{res['name']}")

                        if not is_persp:
                            _, tuned_figs = particle_analysis(
                                None, [], res['ratio'], "", 
                                save=False, unit=cur_unit, 
                                df_input=res['df'],
                                user_bins=bin_val, 
                                user_x_limit=x_lim,
                                user_y_limit=y_lim
                            )
                        else:
                            tuned_figs = build_perspective_figures(res['df'], unit=cur_unit)

                        if "dist" in tuned_figs:
                            tuned_figs["dist"].update_layout(height=500) 
                            st.plotly_chart(tuned_figs["dist"], use_container_width=True)
                            
                            if not res['df'].empty:
                                d50 = np.percentile(res['df'][dia_key], 50)
                                d90 = np.percentile(res['df'][dia_key], 90)
                                col1, col2 = st.columns(2)
                                col1.metric(f"Median", f"{d50:.2f} {cur_unit}")
                                col2.metric(f"Top Size", f"{d90:.2f} {cur_unit}")
                        st.markdown("---")
                    with tab4:
                            from analysis import generate_publication_figure
                            fig = generate_publication_figure(res['df'], save=False)
                            if fig is not None:
                              st.pyplot(fig)

        # -----------------------------
        # Clustering and dataset sorting
        # -----------------------------
        if st.session_state.dataset_sorter is not None and st.session_state.dataset_sorter.summary_df is not None:
            sorter_df = st.session_state.dataset_sorter.summary_df
            if not sorter_df.empty:
                st.divider()
                st.subheader("Automated Process Condition Discovery Deck")
                
                valid_rows = sorter_df[sorter_df["particle_count"] > 0]
                unique_sizes = valid_rows["avg_size_nm"].nunique() if not valid_rows.empty else 0

                if len(valid_rows) <= 1 or unique_sizes == 1:
                    if len(valid_rows) == 1:
                        st.info("💡 **Single Micrograph Detected:** Bypassing clustering controls. directly sort this sample into its designated category folder below.")
                    else:
                        st.info("💡 **Identical Particle Sizes Detected:** All valid micrographs share the exact same mean size. Bypassing clustering to group them uniformly.")
                    
                    if not valid_rows.empty:
                        min_sz = valid_rows["avg_size_nm"].min()
                        updated_manifest = st.session_state.dataset_sorter.process_live_grouping(
                            n_clusters=1, target_min=min_sz - 1, target_max=min_sz + 1
                        )
                    else:
                        updated_manifest = sorter_df.copy()
                        updated_manifest["group_category"] = "Excluded"
                        updated_manifest["status"] = "Excluded"

                else:
                    st.markdown("Categorize recipe groups or filter microstructures into structured physical folders using your final high-fidelity metrics.")
                    
                    ctrl_col1, ctrl_col2 = st.columns(2)
                    with ctrl_col1:
                        num_clusters = st.slider(
                            "Number of Process Conditions (Clusters)", 
                            min_value=1, 
                            max_value=min(5, len(valid_rows)), 
                            value=min(3, len(valid_rows)), 
                            step=1
                        )
                    with ctrl_col2:
                        min_detected_size = int(valid_rows["avg_size_nm"].min())
                        max_detected_size = int(sorter_df["avg_size_nm"].max() + 25)
                        
                        target_range = st.slider(
                            "Filter Micrographs by Size Window (nm)", 
                            min_value=max(1, min_detected_size - 10), 
                            max_value=max_detected_size, 
                            value=(min_detected_size, max_detected_size)
                        )

                    updated_manifest = st.session_state.dataset_sorter.process_live_grouping(
                        n_clusters=num_clusters, 
                        target_min=target_range[0], 
                        target_max=target_range[1]
                    )

                st.markdown("##### Discovery Insights Summary")
                num_valid_images = len(updated_manifest[updated_manifest["status"] == "Ready For SEM Analysis"])
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(label="Active Recipe Groups", value="1" if (len(valid_rows) <= 1 or unique_sizes == 1) else f"{num_clusters}")
                with m2:
                    st.metric(label="Micrographs in Focus Window", value=f"{num_valid_images} / {len(sorter_df)}")
                with m3:
                    active_groups = updated_manifest[updated_manifest["status"] == "Ready For SEM Analysis"]
                    if not active_groups.empty:
                        dominant_raw = active_groups["group_category"].mode()[0]
                        dominant_clean = dominant_raw.split(" (~")[0] if " (~" in dominant_raw else dominant_raw
                        st.metric(label="Dominant Recipe Deck", value=dominant_clean)
                    else:
                        st.metric(label="Dominant Recipe Deck", value="None (Filtered)")

                st.markdown("---")
                st.markdown("#### Clustering Log")
                st.dataframe(
                    updated_manifest[["file_name", "particle_count", "display_label", "group_category"]],
                    use_container_width=True,
                    column_config={
                        "file_name": "Micrograph Name", 
                        "particle_count": "Accurate Particle Count",
                        "display_label": "True Size (Mean ± σ)", 
                        "group_category": "Assigned Cluster Category"
                    }
                )

                if st.button("Save Sorted Categories into Folders", use_container_width=True):
                    save_progress = st.empty()
                    save_progress.info("Sorting file pathways and deploying structural groupings onto output storage...")
                    
                    raw_images_map = {res["name"]: res["original_img"] for res in st.session_state.processed_data}
                    
                    for _, row in updated_manifest.iterrows():
                        raw_label = str(row["group_category"])
                        unified_grp_name = raw_label.split(" (~")[0] if " (~" in raw_label else raw_label
                        
                        sanitized_grp = unified_grp_name.replace("/", "_").replace(" ", "_").replace("~", "").replace("(", "").replace(")", "")
                        cluster_folder = OUT_BASE_DIR / sanitized_grp
                        cluster_folder.mkdir(exist_ok=True)
                        
                        f_name = row["file_name"]
                        f_stem = Path(f_name).stem
                        src_analysis_folder = OUT_BASE_DIR / f_stem
                        
                        if src_analysis_folder.exists() and src_analysis_folder.is_dir():
                            dest_analysis_folder = cluster_folder / f_stem
                            if dest_analysis_folder.exists():
                                shutil.rmtree(dest_analysis_folder)
                            
                            shutil.copytree(src_analysis_folder, dest_analysis_folder)
                            
                            if f_name in raw_images_map:
                                orig_img_rgb = raw_images_map[f_name]
                                orig_dest_path = dest_analysis_folder / f"{f_stem}_original.png"
                                cv2.imwrite(str(orig_dest_path), cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2BGR))
                            
                    save_progress.success("Successfully reorganized analysis output directories into consolidated category decks!")
                    
    elif uploaded_files and st.session_state.roi_mode != "NAVIGATE":
        st.subheader("Define the Region Of Interest")
        ref_path = os.path.join("/tmp", uploaded_files[0].name)
        with open(ref_path, "wb") as f: f.write(uploaded_files[0].getbuffer())
        ref_img, _, _ = load(ref_path)
        h0, w0 = ref_img.shape[:2]
        scale = 900 / w0
        bg_pil = Image.fromarray(cv2.resize(ref_img, (900, int(h0*scale))))
        
        mode_map = {"RECT": "rect", "CIRCLE": "circle"}
        
        canvas_res = st_canvas(
            fill_color="rgba(0, 255, 255, 0.2)",
            stroke_width=2,
            stroke_color="#00ffff",
            background_image=bg_pil,
            height=int(h0*scale),
            width=900,
            drawing_mode=mode_map.get(st.session_state.roi_mode, "rect"),
            key=f"canvas_{st.session_state.canvas_key}"
        )
        st.session_state.roi_json = canvas_res.json_data