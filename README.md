# Project Proteus — ProteusSegNet Application  
### Mapping Suspected Marine Oil Slicks Using SAR & GeoAI

Project Proteus is a research and educational prototype that detects and maps **suspected marine oil slicks** in Synthetic Aperture Radar (SAR) imagery using:

- **SAR physics–based pseudo‑labelling**, and  
- A custom convolutional neural network, **ProteusSegNet**.

The project was developed around the **GEM No.2 oil spill** case study as part of the **National Space Academy 2026 – Earth Observation Application Development (Level 2)** course.

> ⚠ **Important:** All outputs indicate **suspected** oil, not confirmed spills.  
> Labels are derived from SAR heuristics and ESA WorldCover reference data, not human‑annotated ground truth.

---

## Team & Roles

**Team Proteus**

- **Mariolino Daniel III Julabar** – Project lead & primary developer (SAR processing, ProteusSegNet design, app implementation)  
- **AlReem AlRowahi** – Application design, testing, and analysis support  
- **Sultan AlShehhi** – Application design, testing, and analysis support  

Advisory and feedback: **Dr. Vincent Markiet** (National Space Academy EO faculty).

---

## Alignment with UN SDGs & UN Water Conference 2026

Project Proteus was developed **in anticipation of the UN Water Conference in Abu Dhabi in December 2026**, as a demonstration of how Earth Observation and GeoAI can support evidence‑based water and marine‑environment governance.

The project contributes to several **UN Sustainable Development Goals (SDGs)**:

- **SDG 14 – Life Below Water**
  - Early identification of **suspected marine oil slicks** supports efforts to:
    - Protect marine ecosystems from pollution  
    - Inform response and monitoring strategies  
    - Improve situational awareness for coastal and offshore management  

- **SDG 6 – Clean Water and Sanitation**
  - Although focused on the marine environment, Proteus showcases:
    - How EO data can be used for **water‑quality–related surveillance**  
    - The role of satellite monitoring in protecting water resources from hazardous discharges  

- **SDG 13 – Climate Action**
  - Oil spills interact with climate‑driven changes in currents, storms and coastal vulnerability.
  - Systematic satellite monitoring of marine pollution supports:
    - **Climate‑resilient coastal management**  
    - Integration of EO data in climate risk assessments and adaptation planning  

- **SDG 9 – Industry, Innovation and Infrastructure**
  - The project demonstrates:
    - A **novel SAR + GeoAI workflow** for maritime monitoring  
    - How open EO data and custom models can be integrated into analytic infrastructure and future operational services  

Project Proteus is not an operational system, but a **prototype** illustrating how satellite data and GeoAI can underpin SDG‑aligned monitoring and inform discussions at events such as the **UN Water Conference 2026 in Abu Dhabi**.

---

## Problem, Intended Users & Supported Decisions

### Problem

Marine oil spills can be difficult to monitor consistently over large areas. SAR is highly sensitive to sea‑surface roughness and can reveal oil slicks as **darker patches** on the sea surface, but:

- Low‑wind areas, biogenic films, currents, and rain cells can mimic oil.  
- Manual interpretation is resource‑intensive and subjective.  
- Operational teams need fast, repeatable processing with transparent logic.

### Intended Users

- **Earth Observation analysts** and marine surveillance teams  
- **Research & academic users** studying SAR‑based oil detection  
- **Prototyping teams** integrating SAR/GeoAI into maritime monitoring workflows  

### Decisions Supported

Project Proteus supports early‑stage, analyst‑in‑the‑loop decisions such as:

- Where to **prioritize visual review** in SAR scenes (suspected slick locations)  
- Which objects are **larger / darker / higher‑contrast** and may warrant escalation  
- How the spatial extent and distribution of suspected slicks evolve across scenes  

Proteus is **not** intended as an operational spill confirmation tool; it is a **screening and research prototype**.

---

## Data & Technical Approach

### Input Data

1. **SAR Imagery**
   - Single‑band, calibrated SAR backscatter in **dB**  
   - Georeferenced **GeoTIFF** (CRS required)  
   - Typical use case: marine scenes around GEM No.2, but the app is sensor‑agnostic  
   - Examples of suitable sources (not bundled in this repo):
     - ESA **Sentinel‑1** (Copernicus open data)  
     - Other calibrated SAR missions (according to user licence)  

2. **ESA WorldCover 2021 Reference**
   - Local ESA **WorldCover 10 m 2021 v200** GeoTIFF tiles  
   - Used as a **land / water / unknown** mask:
     - 80 = water; other valid classes = non‑water; 0 = unknown  
   - Required filenames (per tile), e.g.:  
     `ESA_WorldCover_10m_2021_v200_N24E054_Map.tif`  
   - Tiles are **not downloaded** by the app; you must provide them locally.

3. **ProteusSegNet Model Checkpoint**
   - Trained model file: `proteus_model.pt`  
   - Contains:
     - `ProteusSegNet` weights  
     - Normalisation settings (e.g., dB clipping range)  
     - Patch size and training metadata  

> This repository ships **no SAR imagery** and **no WorldCover tiles**.  
> Users are responsible for acquiring data and respecting licence terms (e.g., Copernicus open data licences).

### Workflow Overview

Project Proteus implements a three‑stage workflow:

1. **Pre‑flight Input Checks**
   - Validate SAR:
     - Single band  
     - CRS present  
     - Pixel count  
   - dB sanity check (sample percentiles)  
   - Estimate required WorldCover tiles and report **missing coverage**  
   - Decide processing mode:
     - **Full scene in memory** (≤ 16 million pixels), or  
     - **Tiled processing** with seam‑aware merging (up to ~1.5 billion pixels)

2. **Physics‑Based Pseudo‑Labeling**
   - Align WorldCover tiles to SAR grid  
   - Create a **water mask** from WorldCover (non‑water & unknown excluded)  
   - Compute **local contrast** (51×51 window) vs nearby valid water  
   - Apply SAR physics criteria (per operational mode):
     - Seed threshold in dB  
     - Minimum local contrast  
     - Region growth allowance in dB  
   - Remove small connected components (< 20 pixels)  
   - Output:
     - Labels: `0 = evaluated non‑oil`, `1 = suspected oil`, `255 = excluded`  
     - Scene‑level metrics (estimated area, object counts, confidence categories)

3. **ProteusSegNet Inference & Fusion**
   - Patch‑wise model inference (default patch size **256×256**)  
   - Fixed SAR normalisation:
     - Clip to e.g. −40 to +5 dB  
     - Linear scaling to [0, 1]  
   - Combine physics and model:
     - Keep pixels that:
       1. Are within valid water and pass physics criteria, and  
       2. Have model score ≥ configurable threshold (e.g., 0.5)  
     - Remove objects < 20 pixels  
   - Compute object‑level statistics:
     - Area (km²) when CRS geometry allows  
     - Mean backscatter and contrast vs evaluated background water  
     - Mean model score  
     - Heuristic confidence level (high / medium / low / unknown)  

### Tiled Processing (Large Scenes)

For scenes above **16 million pixels**, Proteus uses a tiled engine that:

- Reads the SAR image in **tiles** (default **2048 px** with **32 px halo**)  
- Computes local contrast and region growth **per tile**  
- Merges regions across tile seams using **connected‑component analysis**  
- Produces results equivalent to full‑scene processing while:
  - Limiting peak RAM (depends on tile size, not scene size)  
  - Using disk‑backed scratch label maps (≈ 5 bytes per pixel)  
- Supports scenes up to **~1.5 billion pixels**, subject to available disk space  

### Operational Modes & Key Parameters

**Operational modes** (physics presets in `OPERATIONAL_MODES`):

- **Conservative**  
  Stricter (darker) threshold; fewer, higher‑contrast detections  

- **Balanced** (default)  
  Used in GEM No.2 demonstrations  

- **Sensitive**  
  Captures fainter, thinner slicks, at the cost of more look‑alikes  

Each mode modifies three physics parameters:

- `threshold_offset_db` – offset applied to base dB threshold  
- `contrast_min` – minimum local contrast relative to nearby water  
- `growth_allowance_db` – maximum dB above threshold allowed during region growth  

Other key parameters:

- **Patch size (model)**: 256 px  
- **Minimum object size**: 20 pixels  
- **Model threshold**: default 0.5 (configurable 0.05–0.95)  
- **Zero‑exclusion**: optionally exclude zero‑valued SAR pixels  

---

## Requirements

### Notebook (`ProteusSegNet.ipynb`)

A minimal environment sufficient for the notebook:

```text
numpy
torch
rasterio
scipy
pyproj
matplotlib
gradio>=5
```

The notebook includes a setup cell that installs missing libraries when run in Google Colab.

### Desktop Application

The desktop app uses its own environment (see `desktop_app/requirements.txt`).

---

## Installation & How to Run

You can run Proteus in two main ways:

1. **Google Colab notebook (Gradio app)** – recommended and easiest  
2. **Local desktop application (Windows, Python)** – for offline use  

### 1. Google Colab (recommended)

#### Requirements

- Google account  
- Modern browser  
- Colab runtime (CPU is sufficient; GPU (e.g. T4) optional)  
- Python version managed by Colab (Python 3.x)  

#### Steps

1. **Open the notebook**

   - GitHub: `ProteusSegNet.ipynb`  
   - Or directly in Colab:  
     `https://colab.research.google.com/github/mariolinoiii/project-proteus/blob/main/ProteusSegNet.ipynb`

2. **Select runtime**

   - In Colab: `Runtime → Change runtime type → Python 3`  
     GPU (`T4`) is optional.

3. **Run the notebook**

   - In Colab: `Runtime → Run all`  
   - The setup cell installs any missing libraries:
     - `numpy`, `torch`, `rasterio`, `scipy`, `pyproj`, `matplotlib`, `gradio>=5`

4. **Wait for the Gradio app**

   - The final cell launches a Gradio web UI inside Colab, with authentication.
   - In this repository’s notebook, authentication is configured as:
     ```python
     proteus_app.launch(
         share=True,
         inline=True,
         prevent_thread_lock=True,
         auth=("poseidon", "n5@202613v3!2"),
         ...
     )
     ```
   - These credentials are **demo credentials for this course prototype** only.  
     Do not reuse this username/password pair for any other system.

5. **Sign in to the app**

   - Username: `poseidon`  
   - Password: `n5@202613v3!2`

6. **Upload inputs (Analysis Workspace tab)**

   - **SAR imagery**: single‑band, calibrated GeoTIFF in dB  
   - **WorldCover tiles**: ESA WorldCover GeoTIFFs with original filenames  
   - **Model checkpoint**: `proteus_model.pt` (optional if a default is already present under `/content/proteus_models`)

7. **Configure detection settings**

   - Choose **Operational sensitivity** (Conservative / Balanced / Sensitive)  
   - Adjust **Base backscatter threshold (dB)** if needed  
   - Set **Model score threshold**  
   - Advanced:
     - Zero‑value exclusion  
     - Device: CPU or Auto (uses GPU if available)  
     - Processing strategy: Auto / Always tiled / Full scene (≤16M px)  
     - Tile size: 1024 / 2048 / 4096 px  

   A live summary of the **effective physics parameters** is shown.

8. **Execute analysis**

   - Click **“▶ Execute analysis”**.  
   - The progress bar tracks physics, model inference, object analysis and export.

9. **Review and download outputs**

   - KPI cards with area, object counts, water coverage, runtime, device  
   - Map, backscatter histogram, object statistics plots  
   - Largest objects table  
   - `report.json` and raw JSON view  
   - Download:
     - Full analysis ZIP (rasters + vectors + figures + report)  
     - Individual files  

#### Expected Runtime

- Small scenes (≤ 16 Mpx, CPU): typically **1–5 minutes**  
- Large tiled scenes (100–500 Mpx, CPU): **several minutes**, depending on hardware and tile size  
- GPU primarily accelerates the ProteusSegNet inference stage.

> Each new Colab runtime is temporary. You must re‑upload **model checkpoints**, **SAR**, and **WorldCover** files when you reconnect.

---

### 2. Local Desktop Application (Windows)

A standalone desktop app mirroring the Colab UI is included under `desktop_app/`.

#### Requirements

- Windows 10/11  
- Python 3.10+ (recommended)  
- Local SAR and WorldCover GeoTIFF files  

#### Setup & Launch

From a terminal (PowerShell / CMD):

```bash
# Clone the repository
git clone https://github.com/mariolinoiii/project-proteus.git
cd project-proteus/desktop_app

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the desktop app
python app.py
```

Then:

1. Open the local web UI in your browser (URL printed by the app, usually `http://127.0.0.1:7860`).  
2. Load SAR GeoTIFFs and WorldCover tiles.  
3. Provide or select a `proteus_model.pt` checkpoint.  
4. Configure detection settings and run analysis.  
5. Review results and export rasters, vectors, and reports.

> The desktop app does not require Colab or an internet connection, but you must supply local SAR and WorldCover data.

---

## Sample Data & Reproducible Example

This repository now includes **sample input data** and a **reproducible example output** to demonstrate the full pipeline.

### `data/` and `data/sample_input/`

- `data/ProteusSegNet_Data_Acquisition.pdf`  
  - Documentation on how the SAR and reference data for Proteus were acquired.  

- `data/ProteusSegNet_Demonstration.pdf`  
  - Demonstration case description for the GEM No.2‑like scenario.  

- `data/sample_input/`  
  - `SAR_20261027_Subset-2.zip`  
    - Contains a subset SAR scene `SAR_20261027_Subset-2` (single‑band, calibrated in dB).  
  - `ESA_WorldCover_10m_2021_v200_N24E054_Map.zip`  
    - Contains the matching WorldCover tile `ESA_WorldCover_10m_2021_v200_N24E054_Map.tif`.  

You can unzip these archives and use the contained files as **inputs** to Proteus in either Colab or the Windows app.

### `results/` – Example Output

- `results/Proteus_results_20261027_Conservative_score0.8/`  
  Contains a complete output set generated from the sample input using:
  - Mode: **Conservative**  
  - Model score threshold: **0.8**

Files included:

- `backscatter_histogram.png`  
- `model_scores.tif`  
- `object_statistics.png`  
- `oil_object_ids.tif`  
- `oil_objects.csv`  
- `oil_objects.geojson`  
- `oil_preview.png`  
- `report.json`  
- `suspected_oil_labels.tif`  

These outputs demonstrate the expected structure and content of a Proteus analysis. You can:

1. Run the notebook or app using the sample input from `data/sample_input/`.  
2. Use the same settings (Conservative mode, score threshold 0.8).  
3. Compare your outputs to `results/Proteus_results_20261027_Conservative_score0.8/` to verify reproducibility.  

---

## Example Input, Output, Results & Limitations

### Example Input (GEM No.2‑like Scene)

- **SAR**: Single Sentinel‑1 or equivalent SAR scene (VV or HH), calibrated to σ⁰ in **dB**  
- **Size**: ~4000 × 4000 pixels (16 Mpx, full‑scene mode)  
- **CRS**: UTM zone or geographic CRS  
- **WorldCover**: All tiles covering the SAR extent, e.g.:
  - `ESA_WorldCover_10m_2021_v200_N24E054_Map.tif`  
- **Model**: `proteus_model.pt` trained on physics‑generated pseudo‑labels  

### Example Output Contents (Analysis ZIP)

- **Rasters (GeoTIFF)**  
  - `suspected_oil_labels.tif`  
  - `model_scores.tif`  
  - `oil_object_ids.tif`  

- **Vectors & Tables**  
  - `oil_objects.geojson`  
  - `oil_objects.csv`  

- **Reports & Figures**  
  - `report.json`  
  - `oil_preview.png`  
  - `backscatter_histogram.png`  
  - `object_statistics.png`  

See `results/Proteus_results_20261027_Conservative_score0.8/` for a concrete example.

### Typical Results (Qualitative)

- Regions of **dark, coherent structures** on the sea surface aligned with:
  - Known GEM No.2 slicks (in the case study), and
  - Potential look‑alikes (low‑wind zones, biogenic films, etc.)  
- Higher confidence (high area & strong contrast) assigned to:
  - Larger, darker objects distinctly below background water backscatter  
- Lower confidence assigned to:
  - Small, marginally darker, or low‑contrast objects  

### Key Limitations

- **Not operational**:
  - No independent validation on a human‑annotated test set yet  
  - Predictions must be **reviewed by experienced analysts**  

- **Label quality**:
  - Training labels are **physics‑generated pseudo‑labels**, not ground truth  
  - Physics rules and WorldCover classes introduce systematic biases  

- **False positives / false negatives**:
  - Low‑wind areas, biogenic films, and rain cells can be mis‑detected as oil  
  - Thin or very low‑contrast slicks can be missed, especially in high sea states  

- **Model confidence**:
  - Model scores are **not calibrated probabilities**  
  - Heuristic confidence (high / medium / low / unknown) is based on contrast and area only  

- **Reference data limitations**:
  - WorldCover is a general land‑cover map, not a specialised marine mask  
  - Areas without WorldCover coverage remain excluded from analysis  

- **Geometry & area estimates**:
  - Area (km²) is computed from CRS geometry and pixel size  
  - Values are **indicative**, not official spill measurements  

- **Resource constraints**:
  - Tiled processing writes temporary files (~5 bytes/pixel) to disk  
  - Large scenes require sufficient free disk and RAM  

---

## Repository Structure

```text
project-proteus/
├── ProteusSegNet.ipynb                     # Main Google Colab + Gradio application
├── requirements.txt                        # Notebook environment requirements
├── data/
│   ├── ProteusSegNet_Data_Acquisition.pdf  # Data acquisition document
│   ├── ProteusSegNet_Demonstration.pdf     # Demonstration case description
│   └── sample_input/
│       ├── SAR_20261027_Subset-2.zip
│       └── ESA_WorldCover_10m_2021_v200_N24E054_Map.zip
├── results/
│   └── Proteus_results_20261027_Conservative_score0.8/
│       ├── backscatter_histogram.png
│       ├── model_scores.tif
│       ├── object_statistics.png
│       ├── oil_object_ids.tif
│       ├── oil_objects.csv
│       ├── oil_objects.geojson
│       ├── oil_preview.png
│       ├── report.json
│       └── suspected_oil_labels.tif
├── desktop_app/
│   ├── app.py                              # Windows desktop entry point (Gradio/GUI)
│   ├── requirements.txt                    # Desktop-specific dependencies
│   └── ...                                 # Additional scripts and resources
└── README.md                               # This document
```

> The **notebook** and the **desktop app** use separate environments.  
> Do **not** merge their requirements into a single environment.

---

## Methodology & Validation Status

### Summary of Processing Steps

1. **Input validation** (single band, CRS, dB sanity, pixel count)  
2. **WorldCover alignment** and water‑mask construction  
3. **SAR physics pseudo‑labelling**:
   - dB threshold, local contrast, region growth, small object removal  
4. **ProteusSegNet inference**:
   - Patch‑wise CNN segmentation on normalised SAR  
5. **Fusion**:
   - Keep pixels that satisfy both physics and model thresholds  
6. **Object analysis**:
   - Area, backscatter statistics, contrast, mean model score, heuristic confidence  
7. **Export**:
   - GeoTIFF rasters, GeoJSON & CSV layers, JSON report, PNG figures  

### Validation Status

- No dedicated, human‑annotated test set available yet  
- No formal accuracy metrics (IoU, precision, recall) reported  
- Current use is **exploratory and educational**; outputs are hypotheses for human review  

Planned work (roadmap):

- Create a manually labelled validation subset for GEM No.2  
- Quantitatively compare:
  - Physics‑only detections, vs  
  - Physics + ProteusSegNet detections  
- Evaluate generalisation to other spills and non‑spill SAR scenes  

---

## Security & Data Handling

- Do **not** reuse the demo Gradio credentials (`poseidon` / `n5@202613v3!2`) for any other system.  
- Do **not** commit personal or production usernames/passwords or API keys.  
- All processing occurs on user‑supplied SAR and WorldCover files; the repository does not distribute imagery.

---

## AI‑Assisted Development

The project made use of AI tools (e.g., ChatGPT, Gemini, gIQ AI Assistant, Perplexity) for:

- Code generation and refactoring  
- Debugging and performance improvements (e.g., tiled processing)  
- Drafting documentation and usage guidelines  

All architectural choices, final code, and interpretation remain the responsibility of **Team Proteus**.

---
