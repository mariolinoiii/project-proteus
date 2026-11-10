# Project Proteus / ProteusSegNet — Windows Local Application

Team: **Mariolino Daniel III Julabar, AlReem AlRowahi, Sultan AlShehhi**  
Application: **Project Proteus – Mapping of Suspected Oil Slicks using SAR & GeoAI (Windows desktop)**

This folder contains the **standalone Windows application** for Project Proteus. It runs entirely on your local machine using a Python virtual environment, Gradio web UI, and the same core processing pipeline as the Colab notebook:

- Physics‑based SAR pseudo‑labelling
- ProteusSegNet CNN inference
- Seam‑aware tiled processing for large scenes
- Interactive analysis workspace, dataset preparation, and optional training

---

## Quick Start (Windows)

1. **Extract the ZIP**
   - Right‑click the downloaded ZIP → **Extract All…**  
   - Do **not** run the BAT files from inside the ZIP.

2. **Open the application folder**
   - Navigate into the extracted `ProteusSegNet` (or `desktop_app`) folder containing:
     - `app.py`
     - `Setup_Windows.bat`
     - `Start_Proteus.bat`
     - `models/`, `worldcover_cache/`, `outputs/`, `requirements.txt`, `login.json` (optional)

3. **Place your model checkpoint**
   - Copy your trained **ProteusSegNet** checkpoint to:
     - `models/proteus_model.pt`
   - Model weights are **not** shipped with this app.

4. **Place ESA WorldCover tiles**
   - Copy **ESA WorldCover 10m 2021 v200** GeoTIFFs into:
     - `worldcover_cache/`
   - Keep the **original filenames**, e.g.  
     `ESA_WorldCover_10m_2021_v200_N24E054_Map.tif`

5. **Run setup (one‑time)**
   - Double‑click `Setup_Windows.bat`
   - This will:
     - Detect **64‑bit Python 3.11 or 3.12**
     - Create a virtual environment `.venv`
     - Install all Python dependencies from `requirements.txt`
     - Verify imports for `torch`, `rasterio`, `scipy`, `pyproj`, `matplotlib`, `gradio`
   - If Python is missing, you’ll see instructions to install it. The setup script **does not** install Python itself.

6. **Start the application**
   - Double‑click `Start_Proteus.bat`
   - Keep its terminal window open while you use the app.
   - By default, the app runs at:
     - `http://127.0.0.1:7860`

7. **Browser access**
   - Your default browser should open automatically.
   - If not, manually open: `http://127.0.0.1:7860` in your browser.

8. **Sign‑in (optional)**
   - If `login.json` or environment variables are set, the app will require a username and password.  
   - Example demo credentials (if configured in `login.json`):
     - **Username:** `poseidon`  
     - **Password:** `n5@202613v3!2`
   - See **Login and privacy** below for how to configure this.

9. **Run an analysis**
   - In the **Analysis Workspace** tab:
     - Upload your **calibrated single‑band SAR GeoTIFF in dB**
     - Confirm the **Input check** panel (size, CRS, dB sanity, WorldCover coverage, processing plan)
     - Optionally upload additional WorldCover tiles and/or a different `proteus_model.pt`
     - Choose detection settings and click **“▶ Execute analysis”**
   - Review the figures, KPIs, tables, and download the analysis ZIP.

---

## What’s New in This Windows Version (v2)

### Tiled Processing for Large Scenes

- Scenes above **16 million pixels** are processed tile‑by‑tile:
  - Default tile size: **2048 px**
  - Halo: **32 px** (for correct local contrast and region growth)
- Seam‑aware region merging:
  - Slicks crossing tile boundaries are combined into **single objects**
  - Pixel‑wise results are **identical** to full‑scene processing
- Supports scenes up to **~1.5 billion pixels**, limited by disk and RAM
- Peak RAM depends on **tile size**, not total scene size
- Temporary scratch files (~5 bytes per pixel) are written to:
  - The system temp folder, or  
  - A custom folder set via `PROTEUS_SCRATCH_DIR`

### Refined Interface

- **Input check** panel:
  - SAR size, band count, CRS
  - dB sanity check
  - WorldCover coverage and missing tiles
  - Processing plan (full‑scene vs tiled, resource estimate)
- Live **effective parameter** display:
  - Shows mode‑dependent thresholds and growth ranges:
    - Conservative / Balanced / Sensitive
- **Progress bar** and detailed status messages
- **KPI cards**:
  - Suspected‑oil area, object counts, share of evaluated water, runtime and device
- **Analytical figures**:
  - SAR + suspected‑oil map
  - Backscatter distribution histogram
  - Object area vs contrast scatter plot
- **Largest‑object table**:
  - Top 25 objects with area, pixels, mean dB, contrast, mean score, and heuristic confidence
- **Tabbed** navigation:
  - Analysis Workspace
  - Dataset Preparation
  - Model Training
  - User Guide
  - Methodology & Limitations

---

## Prerequisites (Windows Local App)

- **Operating system**
  - 64‑bit Windows 10 or Windows 11
- **Python**
  - 64‑bit Python **3.11** or **3.12**, with the **Python launcher (`py`)**
  - If Python is missing or incompatible, `Setup_Windows.bat` will stop and show instructions
- **Internet connection (first run only)**
  - Required to download Python packages (especially **PyTorch**)
- **Hardware**
  - CPU is sufficient (default inference device)
  - A CUDA‑capable GPU is optional; if installed and `Auto` is selected, it can accelerate inference

> This is a Python application with a double‑click launcher, **not** a single EXE.  
> The `.venv` folder contains the self‑contained Python environment used by the app.

---

## Model and Training

- Default model path: `models/proteus_model.pt`
- Place your trained ProteusSegNet checkpoint at that path.
- The app expects a checkpoint with:
  - `"architecture": "ProteusSegNet"`
  - Normalisation metadata (e.g. `clip_min_db`, `clip_max_db`)
  - `patch_size` (typically 256)

### Optional Training (Model Training Tab)

- You can **generate pseudo‑label datasets** (Dataset Preparation tab) and **train a new model** (Model Training tab) directly in the app.
- The training pipeline:
  - Uses physics‑generated pseudo‑labels as targets
  - Performs patch‑wise training with weighted BCE loss
  - Saves the resulting checkpoint to:
    - `models/proteus_model.pt`
- **Warning:** Training **overwrites** any existing default checkpoint.  
  Back up `models/proteus_model.pt` before training.

---

## WorldCover Handling

- **Local cache**:
  - ESA WorldCover tiles for 2021 (10 m, v200) should be placed in:
    - `worldcover_cache/`
  - The app never downloads tiles automatically; it only uses **local files**.
- **Usage in the app**:
  - Analysis Workspace:
    - Reuses cached tiles in `worldcover_cache/`
    - Additional uploads are copied into a working folder and used for that run
  - Dataset Preparation:
    - Uses either uploaded WorldCover tiles or the cache
- **Coverage rules**:
  - Missing tiles → those areas are **excluded**, not extrapolated
  - All non‑water and unknown classes remain excluded
  - WorldCover water class (80) is used as **reference water**, but is not a specialised marine‑only mask

---

## Outputs (Windows App)

- All outputs and analysis ZIPs are stored in:
  - `outputs/` (created automatically if missing)

Each analysis typically produces:

- **GeoTIFF rasters**
  - `suspected_oil_labels.tif`
    - `1` = suspected oil  
    - `0` = evaluated water  
    - `255` = excluded (land, unknown, no data)
  - `model_scores.tif`
    - ProteusSegNet score (0–1) on evaluated water; `-9999` elsewhere
  - `oil_object_ids.tif`
    - Unique integer ID per oil object (0 = no oil)
- **Vector & tabular outputs**
  - `oil_objects.geojson`
    - Polygons of oil objects in WGS84 (EPSG:4326)
    - Attributes: area_km2, mean_backscatter_db, contrast_vs_water_db, mean_model_score, heuristic_confidence
  - `oil_objects.csv`
    - Same attributes in CSV format, one row per object
- **Report**
  - `report.json`
    - Parameters (thresholds, mode, model, device)
    - Pixel and area totals
    - Object counts by confidence level
    - Processing strategy (full scene / tiled) and runtime
    - Reference files and limitations
- **Figures**
  - `oil_preview.png` – SAR + suspected oil map
  - `backscatter_histogram.png` – backscatter distribution by class
  - `object_statistics.png` – object area vs contrast

> Figures are displayed directly inside the app. Downloading the ZIP is optional for viewing them, but required if you want to load the outputs into GIS or other tools.

Old output folders in `outputs/` can be deleted manually when no longer needed.

---

## Authentication (login.json & Environment Variables)

The Windows app supports **optional local sign‑in**:

- By default, if no credentials are provided:
  - The app runs on `127.0.0.1` without authentication.
- To **enable login**, create `login.json` in the same folder as `app.py`:

```json
{
  "username": "poseidon",
  "password": "n5@202613v3!2"
}
```

- Alternatively, credentials can be set via environment variables:
  - `PROTEUS_USERNAME`
  - `PROTEUS_PASSWORD`

The app will:

- Read `PROTEUS_USERNAME` / `PROTEUS_PASSWORD` if set
- Otherwise, fall back to `login.json` if present
- Pass these credentials into `gradio.Blocks.launch(auth=...)`

> The example username/password **(`poseidon` / `n5@202613v3!2`) are demo credentials for this local prototype only**.  
> Do not reuse them for any other system or public service.

---

## Login, Privacy & Security

- The local server binds to:
  - `127.0.0.1` (loopback only)
- The app **does not** create a public share link (no Gradio `share=True`).
- Authentication, if enabled, is for **local access only**.
- This is **not** a hardened production or multi‑user deployment:
  - Do not expose it directly on the internet without additional security measures (reverse proxy, TLS, access control, etc.).
- Keep:
  - `login.json`
  - Any environment‑variable secrets
  - Model files and outputs  
  in a secure location.

---

## Limitations & Resource Considerations

- **Scene size**
  - ≤ 16 million pixels → full‑scene, in‑memory processing
  - > 16 million and ≤ ~1.5 billion pixels → tiled processing is used
  - > ~1.5 billion pixels → rejected (scene must be cropped)
- **Scratch disk usage**
  - Tiled processing uses scratch arrays (~5 bytes per pixel) in:
    - System temp folder, or  
    - `PROTEUS_SCRATCH_DIR` if set
  - Ensure sufficient free disk space for large scenes
- **RAM usage**
  - Peak RAM depends on **tile size**
  - Use smaller tiles (e.g. 1024 px) on low‑RAM machines
- **Figure statistics**
  - For very large scenes, histogram percentiles and figures are computed from a **downsampled preview**, not full resolution
- **Methodological limitations**
  - Labels are **physics pseudo‑labels**, not human‑verified annotations
  - Independent validation vs ground truth and vs physics‑only baselines is still outstanding
  - Training‑image predictions demonstrate execution, **not** accuracy
  - Scores and heuristic confidence levels are **not calibrated probabilities**
  - Area estimates depend on CRS geometry and represent **approximate extents**, not measured spill volumes
  - Connected objects are not guaranteed to correspond to distinct real‑world spills

---

## Stopping and Restarting the Application

- To **stop**:
  - Focus the `Start_Proteus.bat` terminal window
  - Press `Ctrl + C`
- To **start again**:
  - Double‑click `Start_Proteus.bat` again  
  - Do not start a second instance while the first is running; port `7860` will already be in use.

You can change the port by setting an environment variable before launching:

```bat
set PROTEUS_PORT=7861
Start_Proteus.bat
```

---

## Verification Status (Windows v2)

- v2 has been executed end‑to‑end on **Windows 11** (CPU) with:
  - Python 3.11 / 3.12
  - `torch` 2.x
  - `gradio` 5.x
- Using:
  - `models/proteus_model.pt`
  - Example SAR: `SAR_20261021_Subset-2.tif`
- Observed:
  - **Full‑scene** results reproduce the original notebook report exactly  
    (e.g. ~120.1076 km², 558 objects for the GEM No.2 subset).
  - **Tiled** results are **pixel‑identical** to the full‑scene outputs.
  - A ~45.7 M‑pixel scene completed successfully in tiled mode.
- Dependency versions are controlled by `requirements.txt` but this is **not a lockfile**; small version differences may occur.

> Keep your original notebook, model backup, and any critical analyses until you have independently verified this Windows application in your own environment.

---
