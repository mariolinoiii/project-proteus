# Project Proteus · Mapping of Suspected Oil Slicks with SAR & GeoAI

Project Proteus is a research and educational prototype for detecting and mapping **suspected marine oil slicks** from Synthetic Aperture Radar (SAR) imagery using physics‑based pseudo‑labeling and a custom convolutional neural network, **ProteusSegNet**.

The project was developed in the context of the **GEM No.2 oil spill** as part of the **Space Academy 2026 – Earth Observation Application Development Course (Level‑2)**.

> **Important:** Outputs indicate _suspected_ oil, not confirmed oil spills.  
> Labels are derived from SAR physics heuristics and ESA WorldCover reference data, not from human‑annotated ground truth.

---

## Key Capabilities

Project Proteus implements an end‑to‑end GeoAI pipeline:

- **Data Ingestion**
  - Calibrated, georeferenced, single‑band SAR GeoTIFFs expressed in **dB**
  - Local **ESA WorldCover 2021** GeoTIFF tiles as land/water reference

- **Physics‑Based Pseudo‑Labeling**
  - Backscatter thresholding in dB
  - Local contrast against surrounding water
  - Region growth within water areas
  - Removal of small connected components
  - Outputs:
    - Per‑pixel labels:
      - `0` = evaluated non‑oil  
      - `1` = suspected oil  
      - `255` = excluded
    - Scene‑level statistics (estimated oil area, object counts, confidence categories)

- **Dataset Generation**
  - Builds training datasets from SAR + WorldCover:
    - `images/` (SAR), `masks/` (oil pseudo‑labels), `validity/` (evaluated mask)
    - Per‑scene `report.json`
    - Global `dataset_index.csv` with object and scene metrics
  - Exportable as a ZIP archive

- **Model Development – ProteusSegNet**
  - Lightweight encoder–decoder CNN with skip connections
  - Designed for:
    - **Input**: single‑channel SAR  
    - **Output**: single‑channel oil / non‑oil logits
  - Training:
    - Patch‑based sampling from SAR, masks, and validity masks
    - Consistent SAR normalization (fixed dB clipping, normalized to [0, 1])
    - `BCEWithLogitsLoss` with `pos_weight` for rare positive (oil) pixels
    - Outputs `proteusmodel.pt` checkpoint with model weights and metadata

- **Integrated Inference & Analytics**
  - Combined physics + CNN decision logic:
    - Physics labels (`labels == 1`)
    - Model scores (≥ configurable threshold)
    - Evaluated water mask from WorldCover
  - Object‑level analytics:
    - Estimated oil area (km²)
    - Object counts and size distribution
    - Backscatter statistics and contrast vs surrounding water
    - Mean model scores and heuristic confidence categories
  - Export formats:
    - GeoTIFF: label, score, and object ID rasters
    - GeoJSON: oil objects with attributes
    - CSV: per‑object metrics
    - JSON: processing report
    - PNG: maps, histograms, and scatter plots

- **Deployment**
  - **Google Colab + Gradio** notebook application
  - **Standalone Python desktop application** (Windows/PC)
  - No reliance on external cloud AI APIs or pretrained segmentation backbones in the final version

---

## Repository Structure

```text
project-proteus/
├── ProteusSegNet.ipynb          # Main Google Colab / Gradio notebook
├── requirements.txt             # Notebook environment requirements
├── desktop_app/                 # Standalone desktop application (Windows)
│   ├── app.py                   # Desktop application entry point
│   ├── requirements.txt         # Desktop-specific Python dependencies
│   └── ...                      # Setup scripts and support files
└── README.md                    # This document
```

> **Note:** The notebook and desktop app are separate applications with separate environments.  
> Do **not** combine their requirements into a single environment.

---

## Requirements

### Notebook (`ProteusSegNet.ipynb`)

The root‑level `requirements.txt` is specific to the notebook:

```text
numpy
torch
rasterio
scipy
pyproj
matplotlib
gradio==4.44.7
```

The notebook includes a setup cell that installs missing libraries when run in Google Colab.

### Desktop Application

The desktop version uses its own environment:

```text
desktop_app/requirements.txt
```

Follow the instructions inside `desktop_app/` for installation and usage on Windows.

---

## Data Requirements

To run an analysis, prepare:

1. **SAR Image**
   - Calibrated, georeferenced, single‑band SAR GeoTIFF
   - Pixel values in **dB**
   - Maximum size: **16,000,000 pixels** (e.g., 4000 × 4000)

2. **ESA WorldCover 2021 Reference**
   - Local WorldCover GeoTIFF tiles covering the SAR scene
   - Keep the original WorldCover filenames unchanged
   - Used as land/water reference (not exclusively marine water)

3. **ProteusSegNet Model Checkpoint**
   - Compatible trained checkpoint, typically named `proteusmodel.pt`
   - Must match the expected architecture and normalization settings

> **Important:** Model weights and input imagery are **not** embedded in the notebook.  
> You must supply your own SAR data, reference tiles, and model checkpoint.

---

## Quickstart

### Option 1: Google Colab / Gradio Notebook

1. Open the main notebook:
   - GitHub: `ProteusSegNet.ipynb`
   - Or directly in Colab:  
     `https://colab.research.google.com/github/mariolinoiii/project-proteus/blob/main/ProteusSegNet.ipynb`
2. Connect to a runtime.
3. From the Colab menu, select **Runtime → Run all**.
4. Wait for the Gradio application interface to appear.
5. Sign in using application credentials (provided separately by the project maintainer).
6. Go to the **Analysis Workspace** tab.
7. Upload:
   - SAR GeoTIFF (in dB)
   - Matching local WorldCover reference files
   - A compatible `proteusmodel.pt` checkpoint
8. Review and adjust processing settings as needed.
9. Click **Execute analysis**.
10. Review the report and analytical figures.
11. Download the **analysis ZIP** using the control beneath the figures.

Notes:

- CPU is the default processing option.
- In a fresh Colab runtime, you must upload the model checkpoint again.
- If a default checkpoint already exists in the runtime, the model upload field may be left empty.

### Option 2: Desktop Application (Windows)

1. Navigate to `desktop_app/`.
2. Create and activate a Python environment using `desktop_app/requirements.txt`.
3. Follow the setup instructions provided in that folder.
4. Run the desktop application (e.g., `python app.py`).
5. Use the GUI to:
   - Load local SAR GeoTIFFs and WorldCover tiles
   - Configure analysis parameters
   - Run inference and review/export outputs

The desktop app:

- Does not require Colab or an internet connection
- Manages authentication and configuration locally
- Mirrors core functionality of the Colab app for incident analysis

---

## Workflow

### 1. Analysis Workflow (ProteusSegNet Inference)

From the **Analysis Workspace** tab:

1. **Upload Inputs**
   - SAR GeoTIFF (single band, dB)
   - Matching ESA WorldCover 2021 tiles
   - ProteusSegNet checkpoint (`proteusmodel.pt`), if not already present

2. **Run Processing**
   - Input validation:
     - Single band, georeferenced CRS
     - Pixel count ≤ 16M
   - Physics‑based label generation (`create_physics_labels`):
     - Build reference mask (unknown / non‑water / water)
     - Apply backscatter threshold and local contrast filter
     - Region grow within water and below extended threshold
     - Remove small connected components
   - Model inference:
     - Load ProteusSegNet checkpoint
     - Apply patch‑wise inference over SAR
   - Fusion and post‑processing:
     - Combine physics labels, model scores, and evaluated water mask
     - Remove small objects
     - Generate final label map and object IDs
   - Compute scene and object statistics.

3. **Review Outputs**
   - Inline maps and figures:
     - SAR and suspected‑oil preview
     - Backscatter distribution histogram
     - Object area vs contrast scatter plots
   - Exportable ZIP with rasters, vectors, and reports.

### 2. Optional: Dataset Preparation

From the **Dataset Preparation** tab:

1. Upload a folder of training SAR images.
2. Upload the matching WorldCover reference tiles.
3. Run **Prepare annotated dataset**.
4. Review the dataset preparation report.
5. Download the prepared dataset ZIP, which includes:
   - `images/` – Original SAR GeoTIFFs
   - `masks/` – Binary oil pseudo‑labels (0/1)
   - `validity/` – Valid vs excluded pixels
   - `report.json` per scene
   - `dataset_index.csv` with per‑scene metadata and object statistics

Labels are **pseudo‑labels** generated by SAR physics rules and WorldCover, not manually verified annotations.

### 3. Optional: Model Training

From the **Model Training** tab:

1. Provide a prepared dataset (from the Dataset Preparation step).
2. Set the number of training epochs and other parameters as required.
3. Confirm that you authorize training and **replacement of the default checkpoint**.
4. Run **Start model training**.
5. After training:
   - Download the resulting `proteusmodel.pt` checkpoint.

> **Warning:**  
> The default checkpoint may be overwritten during training.  
> Back up any existing checkpoint before starting a new training run.

---

## Outputs

A typical **analysis ZIP** may contain:

- **Rasters (GeoTIFF)**
  - Suspected‑oil label raster
  - Model score raster
  - Oil‑object ID raster

- **Vectors & Tables**
  - Object polygons in **GeoJSON**
  - Object statistics in **CSV**

- **Reports & Figures**
  - Processing report in **JSON**
  - SAR and suspected‑oil preview maps (**PNG**)
  - Backscatter distribution histogram (**PNG**)
  - Object area vs contrast figure (**PNG**)

Model checkpoints (e.g., `proteusmodel.pt`) are downloaded separately from the **Model Training** tab and are **not** part of the analysis ZIP.

---

## Methodology Overview

### Physics‑Based Pseudo‑Labeling (`create_physics_labels`)

- Validates SAR inputs (size, band count, CRS, dB scale)
- Builds a reference mask from ESA WorldCover:
  - `0` = unknown
  - `1` = non‑water
  - `2` = water
- Applies SAR‑based heuristics:
  - Global backscatter threshold in dB
  - Local contrast measure relative to surrounding water (windowed filtering)
  - Region growth within water at or below an extended threshold
  - Removal of small connected components
- Outputs:
  - Label raster with classes 0/1/255
  - Scene‑level metrics (estimated oil area, object counts, confidence categories)

### Dataset Generation

Core functions (in code):

- `batch_generate_oil_labels`  
- `generate_dataset_from_uploads`

Workflow:

- For each SAR image and its matching WorldCover tiles:
  - Run physics labeler → oil pseudo‑mask + validity mask
  - Copy SAR to `images/`, store masks in `masks/` and `validity/`
  - Generate a per‑scene `report.json`
- Aggregate:
  - `dataset_index.csv` with:
    - `image_id`, `sar_path`, `label_path`, `validity_path`
    - Estimated oil area (km²)
    - Object statistics (counts, size distribution)

### Model Training – ProteusSegNet

- Architecture:
  - Custom encoder–decoder CNN with skip connections
  - Single‑channel input and output (logit for oil / non‑oil)
  - Lightweight design for SAR segmentation

- Training (`train_proteus_model`):
  - Patch‑based sampling from SAR, label, and validity masks
  - Class imbalance handling and checks for oil / non‑oil balance
  - Fixed SAR clipping range (e.g., −40 to +5 dB) and normalization to [0, 1]
  - `BCEWithLogitsLoss` with positive class weighting

- Checkpoint includes:
  - Model weights and architecture identifier
  - Normalization parameters and patch size
  - Training loss history
  - Metadata noting that labels are physics‑based pseudo‑labels

### Inference & Analytics (`run_proteus_project`)

- End‑to‑end workflow:
  1. Input validation (SAR + WorldCover)
  2. Physics label generation
  3. Model loading and patch‑wise inference
  4. Fusion of physics labels, model scores, and evaluated water mask
  5. Post‑processing (size filtering, object re‑labeling)
  6. Object and scene statistics generation
  7. Export of rasters, vectors, and analytical visualizations

---

## Limitations

- **Input Size**
  - Maximum SAR image size: **16,000,000 pixels** per image
  - Automatic tiling / merging of oversized scenes is **not implemented**

- **Validation**
  - No independent, human‑annotated validation dataset yet
  - Demonstrations on training imagery are not independent accuracy tests
  - Planned: manual annotation of GEM No.2 subset and computation of IoU, precision, and recall

- **Model Confidence**
  - Model scores are **not** calibrated probabilities of oil
  - Object confidence categories are heuristic

- **Reference Data**
  - ESA WorldCover water classes are not specifically marine‑water classes
  - Areas without reference coverage are excluded

- **Area Estimates**
  - Area estimates depend on the coordinate reference system geometry
  - They are **not** confirmed spill volumes or official measurements

- **Resource Usage**
  - Processing and large embedded downloads may require substantial runtime and browser memory
  - Notebook performance depends on the available Colab resources

---

## Roadmap (Planned Enhancements)

- **Formal Quantitative Validation**
  - Human‑labeled validation/test set for GEM No.2
  - Metrics for:
    - Physics‑only detections
    - Physics + ProteusSegNet detections

- **Generalization Testing**
  - Evaluation on other spill and non‑spill SAR scenes
  - Characterization of false alarms and missed detections

- **Integration with External Systems**
  - Export Proteus outputs as layers to external platforms (e.g., gIQ)
  - Wrap ProteusSegNet as a reusable model component
  - Explore automated detection, reporting, and notification workflows

- **User Experience Improvements**
  - Additional validation for incorrect inputs (e.g., non‑dB SAR)
  - Better messaging for missing or misaligned WorldCover tiles
  - Optional configuration file to reload parameter presets

---

## Security and Data Handling

- **Do not** publish passwords, API keys, or other credentials in:
  - Notebook code
  - Saved outputs
  - Committed configuration files
- Keep authentication credentials:
  - Outside the committed notebook
  - Shared only with authorized users
- The desktop application includes local authentication handling; do **not** commit sensitive configuration files.

---

## Project Leadership and Contributions

### Original Developer and Project Lead

**Mariolino Daniel III Julabar**  
GitHub: [@mariolinoiii](https://github.com/mariolinoiii)

- Initiated **Project Proteus**
- Developed the initial implementation, including:
  - Optical and SAR imagery acquisition
  - SAR data processing workflow
  - ProteusSegNet model and pipeline
  - Google Colab / Gradio notebook application
  - Standalone Python desktop application

### Additional Team Members

1. **AlReem Saeed**
2. **Sultan AlShehhi**

Individual contributions and responsibilities will be documented as the project develops.
