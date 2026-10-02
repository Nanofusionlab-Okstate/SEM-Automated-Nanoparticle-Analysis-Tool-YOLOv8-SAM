# SEM - Automated Nanoparticle Analysis Tool

Machine-learning framework for quantitative SEM nanoparticle metrology using object detection (YOLOv8) and instance segmentation (Segment Anything / SAM). Companion code for the paper:

> **Machine Learning Integrated Quantitative SEM Analytics Using Object Detection and Instance Segmentation for Nanoparticle Metrology**
> 
> Karishma Begum, Vikas Reddy Paduri, Nagarajan Anna Ramesh Babu, Ritesh Sachan*
> Oklahoma State University, School of Mechanical and Aerospace Engineering
> *Corresponding author: rsachan@okstate.edu
> *(citation details to be updated with journal/DOI upon publication)*

Given a raw SEM micrograph, the tool detects individual nanoparticles, separates physically touching particles, filters out noise and artifacts, calibrates pixel measurements to physical units from the embedded scale bar, and reports particle-level and population-level morphology, spatial organization, and batch-level comparisons — through an interactive web interface.

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Model weights](#model-weights)
- [Usage](#usage)
- [Module overview](#module-overview)
- [Validation](#validation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Pipeline overview

1. **Scale calibration** (`scripts/metrology.py`) — locates the embedded scale bar in the SEM micrograph footer and converts it to a pixel-to-physical-unit (nm/px) calibration ratio, via template matching against known scale-bar labels with an OCR fallback (Tesseract) for unseen labels.
2. **Detection** (`scripts/detector.py`) — a custom-trained YOLOv8 model proposes candidate nanoparticle bounding boxes, which are passed to Meta's Segment Anything Model (SAM, ViT-H) to produce pixel-accurate instance segmentation masks.
3. **De-clustering and filtering** (`scripts/detector.py`) — watershed-based splitting separates masks that correspond to multiple physically touching particles; an adaptive-threshold rescue stage recovers particles missed by the primary detector; shape-based filters (minimum area, minimum circularity, maximum aspect ratio) remove noise and non-particle artifacts.
4. **Quantitative metrology** (`scripts/analysis.py`) — computes per-particle morphology (equivalent diameter, perimeter, circularity, aspect ratio) and spatial descriptors (nearest-neighbor distance, FFT-derived characteristic spacing, surface coverage).
5. **Batch-level categorization** (`scripts/clustering.py`) — K-means clustering of analyzed micrographs by mean particle size, for organizing large multi-condition datasets.
6. **Visualization & reporting** (`scripts/sam_visualize.py`, `scripts/analysis.py`, `app.py`) — diameter-coded heatmaps, interactive Plotly dashboards, publication-ready static figures, and exportable PDF reports, all served through a Streamlit interface.

**Perspective Mode** (`scripts/perspective.py`) — a parallel pipeline for tilted/oblique SEM micrographs. It detects particles with a dual-confidence YOLO pass (a primary pass plus a low-confidence recall pass for faint particles, with adaptive-threshold rescue boxes), filters by shape (solidity, circularity, aspect ratio) and SAM mask confidence, resolves occluded/stacked particles by area-ranked front-particle-wins logic, and computes per-particle contact angle, height, and base diameter from spherical-cap geometry.

See the paper's Methodology section for the full technical description, matching criteria, and formulas (including the independent validation against ImageJ described below).

## Repository structure

```
.
├── app.py                     # Streamlit application (entry point)
├── scripts/
│   ├──sem_templates/         # Scale-bar label templates used for calibration (add your own as needed)
│   ├── detector.py            # YOLOv8 + SAM detection, de-clustering, filtering
│   ├── data_reader.py         # Image I/O: TIFF / PNG / JPEG / standard formats
│   ├── metrology.py           # Scale-bar calibration (template matching + OCR)
│   ├── analysis.py            # Particle-level & population-level metrology
│   ├── sam_visualize.py       # Mask overlays, heatmaps, annotated figures
│   ├── clustering.py          # Batch-level K-means categorization
│   ├── validator.py           # Pre-screening of micrographs for analysis viability
│   └── perspective.py         # Perspective-mode pipeline: contact angle, height, base diameter for tilted micrographs   
├── outputs/                   # Analysis outputs are written here at runtime (git-ignored)
├── examples/                  # Example input micrographs (optional, add your own)
├── requirements.txt
└── README.md
```

Model weight files (`best12x.pt`, `sam_vit_h_4b8939.pth`) are **not** committed to this repository (see [Model weights](#model-weights)) — they belong at the repository root, alongside `app.py`.

## Installation

```bash
git clone https://github.com/<your-username>/SEM-Automated-Nanoparticle-Analysis-Tool-YOLOv8-SAM.git
cd sem-nanoparticle-analysis-tool

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**System dependency:** the scale-bar calibration's OCR fallback uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), which must be installed separately and available on your `PATH`:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

A CUDA-capable GPU is strongly recommended for the YOLO + SAM detection stage (it will fall back to CPU automatically, but will be considerably slower).

**License note:** this repository's own code is released under the MIT License (see [LICENSE](LICENSE)). It depends on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. AGPL-3.0 is a copyleft license — if you build on or redistribute this pipeline (including as a hosted service), Ultralytics' AGPL terms apply to that dependency, and you may need either to comply with AGPL for your own distribution or obtain a commercial license from Ultralytics for closed-source use. This does not affect academic/research use of this repository as-is.

## Model weights

Two model checkpoints are required and must be placed in the **repository root** (same folder as `app.py`):

| File | Model | Source |
|---|---|---|
| `best12x.pt` | Custom-trained YOLOv8 nanoparticle detector by Genc etl | https://drive.google.com/drive/folders/1-ooqb_eBRD0WLau7fTwLcZzDW7jWfmDM|
| `sam_vit_h_4b8939.pth` | Segment Anything (ViT-H) |https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth|

> `mobile_sam.pt` is referenced as an optional lighter-weight SAM variant for future use and is not required to run the current pipeline.

Download both files and place them directly in the repository root:

```
sem-Automated-Nanoparticle-Analysis-Tool-YOLOv8-SAM/
├── app.py
├── best12x.pt              <-- place here
├── sam_vit_h_4b8939.pth    <-- place here
└── scripts/...
```

## Usage

Launch the interactive application:

```bash
streamlit run app.py
```

This opens a browser-based interface where you can upload SEM micrographs, run detection and metrology, inspect particle-level results and visualizations, and export PDF/CSV reports and publication figures.

## Module overview

| File | Purpose |
|---|---|
| `detector.py` | YOLOv8 candidate detection, SAM segmentation, watershed de-clustering, adaptive-threshold rescue, shape-based filtering |
| `data_reader.py` | Loads SEM images (TIFF, DM3/DM4, and standard formats) and normalizes them to 8-bit RGB |
| `metrology.py` | `SEMCalibrator` class — scale-bar detection via template matching with OCR fallback |
| `analysis.py` | Per-particle morphology and spatial metrology (`particle_analysis`), nearest-neighbor distance, result formatting/export, and the 4-panel publication figure (`generate_publication_figure`) |
| `sam_visualize.py` | Mask overlays, diameter-coded heatmaps, annotated figure rendering |
| `clustering.py` | `DatasetSorter` — K-means batch categorization by mean particle size |
| `validator.py` | Pre-screens micrographs for analysis viability (circularity/solidity checks) before detection |
| `perspective.py` | Perspective-mode pipeline — dual-confidence detection, occlusion resolution, and contact-angle/height/base-diameter metrology for tilted micrographs |
| `app.py` | Streamlit application tying the full pipeline together |

## Validation

The framework's particle detections were independently validated against [ImageJ](https://imagej.net/)'s threshold-based particle analysis on four representative SEM micrographs, using a one-to-one nearest-centroid matching procedure. Pooled across the four micrographs, the framework achieved a precision of 0.97, recall of 0.94, F1 score of 0.95, and a particle-size correlation of 0.94 against ImageJ. Full methodology, matching criteria, and per-image results are reported in the paper's Validation section.

## Citation

If you use this code, please cite:

```bibtex
@article{begum_sem_ml_framework,
  title   = {Machine Learning Integrated Quantitative SEM Analytics Using Object Detection and Instance Segmentation for Nanoparticle Metrology},
  author  = {Begum, Karishma and Paduri, Vikas Reddy and Ramesh Babu, Nagarajan Anna and Sachan, Ritesh},
  journal = {TBD},
  year    = {2026},
  note    = {Manuscript in preparation / under review}
}
```
*(Update with the final journal, volume, page, and DOI once published.)*

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

Questions about this code or the underlying research can be directed to the corresponding author, Ritesh Sachan (rsachan@okstate.edu), Karishma Begum (kbegum@okstate.edu), Vikas Reddy Paduri (vpaduri@okstate.edu) or by opening an issue on this repository.
