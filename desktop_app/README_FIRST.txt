PROJECT PROTEUS / PROTEUSSEGNET — WINDOWS LOCAL APPLICATION
===========================================================

Team: Mariolino Daniel III Julabar, AlReem AlRowahi, Sultan AlShehhi
Application: Project Proteus – Mapping of Suspected Oil Slicks using SAR & GeoAI (Windows desktop)

This folder contains the standalone Windows application for Project Proteus. It runs entirely on your local machine using a Python virtual environment, a Gradio web UI, and the same core processing pipeline as the Colab notebook:

- Physics-based SAR pseudo-labelling
- ProteusSegNet CNN inference
- Seam-aware tiled processing for large scenes
- Interactive analysis workspace, dataset preparation, and optional training


QUICK START (WINDOWS)
---------------------

1. Extract the ZIP
   - Right-click the downloaded ZIP and choose "Extract All…"
   - Do NOT run the BAT files from inside the ZIP.

2. Open the application folder
   - Navigate into the extracted "ProteusSegNet" (or "desktop_app") folder containing:
     - app.py
     - Setup_Windows.bat
     - Start_Proteus.bat
     - models/
     - worldcover_cache/
     - outputs/
     - requirements.txt
     - login.json (optional)

3. Place your model checkpoint
   - Copy your trained ProteusSegNet checkpoint to:
     models/proteus_model.pt
   - Model weights are NOT shipped with this app.

4. Place ESA WorldCover tiles
   - Copy ESA WorldCover 10m 2021 v200 GeoTIFFs into:
     worldcover_cache/
   - Keep the original filenames, for example:
     ESA_WorldCover_10m_2021_v200_N24E054_Map.tif

5. Run setup (one-time)
   - Double-click Setup_Windows.bat
   - This will:
     - Detect 64-bit Python 3.11 or 3.12
     - Create a virtual environment .venv
     - Install Python dependencies from requirements.txt
     - Verify imports for torch, rasterio, scipy, pyproj, matplotlib, gradio
   - If Python is missing, you will see instructions to install it.
   - The setup script does NOT install Python itself.

6. Start the application
   - Double-click Start_Proteus.bat
   - Keep its terminal window open while you use the app.
   - By default, the app runs at http://127.0.0.1:7860

7. Browser access
   - Your default browser should open automatically.
   - If it does not, manually open:
     http://127.0.0.1:7860

8. Sign-in (if login is enabled)
   - If login.json or environment variables are set, the app will require a username and password.
   - Example demo credentials (if login.json contains them):
     Username: poseidon
     Password: n5@202613v3!2
   - See "Login, Privacy & Security" below for how this is configured.

9. Run an analysis
   - In the "Analysis Workspace" tab:
     - Upload your calibrated single-band SAR GeoTIFF in dB
     - Review the "Input check" panel (size, CRS, dB sanity, WorldCover coverage, processing plan)
     - Optionally upload additional WorldCover tiles and/or a different proteus_model.pt
     - Choose detection settings and click "▶ Execute analysis"
   - Review the figures, KPI cards, object table, and download the analysis ZIP.


WHAT’S NEW IN THIS WINDOWS VERSION (v2)
---------------------------------------

Tiled processing for large scenes:
- Scenes above 16 million pixels are processed tile-by-tile:
  - Default tile size: 2048 pixels
  - Halo: 32 pixels (for correct local contrast and region growth)
- Seam-aware region merging:
  - Slicks crossing tile boundaries are combined into single objects
  - Pixel-wise results are identical to full-scene processing
- Supports scenes up to approximately 1.5 billion pixels (limited by disk and RAM)
- Peak RAM depends on tile size, not total scene size
- Temporary scratch files (about 5 bytes per pixel) are written to:
  - The system temp folder, OR
  - A custom folder set via PROTEUS_SCRATCH_DIR

Refined interface:
- Input check panel:
  - SAR size, band count, CRS
  - dB sanity check
  - WorldCover coverage and missing tiles
  - Processing plan (full-scene vs tiled, resource estimates)
- Live effective parameter display:
  - Shows mode-dependent thresholds and growth ranges (Conservative / Balanced / Sensitive)
- Progress bar and detailed status messages
- KPI cards:
  - Suspected-oil area, object counts, share of evaluated water, runtime and device
- Analytical figures:
  - SAR + suspected-oil map
  - Backscatter distribution histogram
  - Object area vs contrast scatter plot
- Largest-object table:
  - Top 25 objects with area, pixel count, mean dB, contrast, mean score, heuristic confidence
- Tabbed navigation:
  - Analysis Workspace
  - Dataset Preparation
  - Model Training
  - User Guide
  - Methodology & Limitations


PREREQUISITES (WINDOWS LOCAL APP)
---------------------------------

Operating system:
- 64-bit Windows 10 or Windows 11

Python:
- 64-bit Python 3.11 or 3.12, with the Python launcher (py)
- If Python is missing or incompatible, Setup_Windows.bat will stop and show instructions.

Internet connection (first run only):
- Required to download Python packages (especially PyTorch)

Hardware:
- CPU is sufficient (default inference device)
- A CUDA-capable GPU is optional; if installed and "Auto" is selected, it can accelerate inference.

Note:
This is a Python application with a double-click launcher, not a single EXE.  
The .venv folder contains the self-contained Python environment used by the app.


MODEL AND TRAINING
------------------

Default model path:
- models/proteus_model.pt

Place your trained ProteusSegNet checkpoint at that path. The app expects a checkpoint that includes:
- "architecture": "ProteusSegNet"
- Normalisation metadata (for example, clip_min_db and clip_max_db)
- patch_size (typically 256)

Optional training (Model Training tab):
- Use the Dataset Preparation tab to generate physics-based pseudo-label datasets (images, masks, validity, dataset_index.csv).
- Use the Model Training tab to:
  - Train ProteusSegNet on that dataset.
  - Save the trained model as models/proteus_model.pt.

Warning:
Training overwrites any existing models/proteus_model.pt.  
Back up the file before training if you want to keep your original weights.


WORLDCOVER HANDLING
-------------------

Local cache:
- ESA WorldCover 10m 2021 v200 tiles must be placed in:
  worldcover_cache/
- The app never downloads tiles automatically; it uses only local files.

Usage:
- Analysis Workspace:
  - Reuses cached tiles in worldcover_cache.
  - Additional WorldCover uploads are copied into a working folder and used for that run.
- Dataset Preparation:
  - Uses uploaded WorldCover tiles or the cache if uploads are empty.

Coverage rules:
- Missing tiles: those areas are excluded (not extrapolated).
- Non-water and unknown classes are excluded.
- WorldCover class 80 (water) is used as reference water, but it is not a strictly marine-only mask.


OUTPUTS (WINDOWS APP)
---------------------

All outputs and analysis ZIPs are stored in:

- outputs/

Each analysis typically produces:

GeoTIFF rasters:
- suspected_oil_labels.tif
  - 1 = suspected oil
  - 0 = evaluated water
  - 255 = excluded (land, unknown, missing reference)
- model_scores.tif
  - ProteusSegNet score (0–1) on evaluated water
  - -9999 elsewhere
- oil_object_ids.tif
  - Unique integer object ID per oil object (0 = no oil)

Vector and tabular outputs:
- oil_objects.geojson
  - Polygons of oil objects in WGS84 (EPSG:4326)
  - Attributes: area_km2, mean_backscatter_db, contrast_vs_water_db, mean_model_score, heuristic_confidence
- oil_objects.csv
  - Same attributes in CSV format, one row per object

Report:
- report.json
  - Parameters (thresholds, mode, model, device)
  - Pixel and area summaries
  - Object counts by confidence class
  - Processing strategy (full scene or tiled) and runtime
  - Reference files and limitations

Figures:
- oil_preview.png
  - SAR plus suspected oil overlay
- backscatter_histogram.png
  - Backscatter distribution (oil vs water)
- object_statistics.png
  - Object area versus contrast

Notes:
- Figures are displayed directly inside the app.
- Downloading the ZIP is optional for viewing figures, but required to load outputs in external tools (e.g. GIS).
- Old output folders in outputs/ can be deleted manually when no longer needed.


AUTHENTICATION (LOGIN.JSON AND ENVIRONMENT VARIABLES)
-----------------------------------------------------

The Windows app supports optional local sign-in:

Default behavior:
- If no credentials are provided:
  - The app runs on 127.0.0.1 without authentication.

To enable login, create login.json next to app.py with content like:

{
  "username": "poseidon",
  "password": "n5@202613v3!2"
}

Environment override:
- You can also configure credentials through environment variables:
  - PROTEUS_USERNAME
  - PROTEUS_PASSWORD

Behavior:
- The app reads PROTEUS_USERNAME / PROTEUS_PASSWORD if they are set.
- Otherwise, it falls back to login.json if present.
- These credentials are passed to Gradio as the auth parameter.

Important:
The example username/password ("poseidon" / "n5@202613v3!2") are demo credentials for this local prototype only.  
Do not reuse them for any other system or public service.


LOGIN, PRIVACY AND SECURITY
---------------------------

- The server binds only to 127.0.0.1 (loopback).
- The app does NOT create a public share link.
- Authentication, if enabled, is for local access only.
- This application is NOT a hardened production or multi-user service:
  - Do not expose it directly on the internet without additional security (reverse proxy, TLS, access control, etc.).
- Keep the following private:
  - login.json
  - Any environment-variable secrets
  - Model files and outputs


LIMITATIONS AND RESOURCE CONSIDERATIONS
---------------------------------------

Scene size:
- Up to 16 million pixels:
  - Full-scene, in-memory processing.
- Above 16 million and up to about 1.5 billion pixels:
  - Tiled processing is used automatically (or when "Always tiled" is selected).
- Above approximately 1.5 billion pixels:
  - The scene is rejected; it must be cropped.

Scratch disk usage:
- Tiled processing uses scratch arrays (about 5 bytes per pixel) in:
  - The system temp folder, or
  - A custom folder specified by PROTEUS_SCRATCH_DIR.
- Ensure sufficient free disk space for large scenes.

RAM usage:
- Peak RAM usage depends on tile size.
- Use smaller tiles (e.g. 1024 px) on low-RAM machines.

Figure statistics:
- For very large scenes, histogram percentiles and figures are computed from a downsampled preview, not full resolution.

Methodological limitations:
- Labels are physics-generated pseudo-labels, not human-verified spill annotations.
- Independent validation against ground truth and a physics-only baseline is still outstanding.
- Training-image predictions demonstrate execution, not accuracy.
- Model scores and heuristic confidence levels are not calibrated probabilities.
- Area estimates depend on CRS geometry and represent approximate extents, not measured spill volumes.
- Connected objects are not guaranteed to correspond to distinct real-world spills.


STOPPING AND RESTARTING THE APPLICATION
---------------------------------------

To stop:
- Focus the Start_Proteus.bat terminal window.
- Press Ctrl + C.

To start again:
- Double-click Start_Proteus.bat.
- Do not start a second instance while the first is running; port 7860 will already be in use.

Changing the port:
- You can change the server port by setting PROTEUS_PORT before launching, for example:

  set PROTEUS_PORT=7861
  Start_Proteus.bat


VERIFICATION STATUS (WINDOWS v2)
--------------------------------

- v2 has been executed end-to-end on Windows 11 (CPU) with:
  - Python 3.11 / 3.12
  - torch 2.x
  - gradio 5.x

- Using:
  - models/proteus_model.pt
  - Example SAR: SAR_20261021_Subset-2.tif

Observed:
- Full-scene results reproduce the original notebook report exactly (for example, about 120.1076 km², 558 objects for a GEM No.2 subset).
- Tiled results are pixel-identical to full-scene outputs.
- A ~45.7 million-pixel scene completed successfully in tiled mode.

Note:
- Dependency versions are controlled by requirements.txt but this is not a strict lockfile; small version differences may occur.
- Keep your original notebook, model backup, and any critical analyses until you have independently verified this Windows application in your own environment.
