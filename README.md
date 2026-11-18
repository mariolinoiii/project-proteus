# Project Proteus — ProteusSegNet Application
### Mapping of Suspected Oil Slicks Using SAR & GeoAI

Project Proteus is a research and educational prototype that identifies and maps suspected marine oil slicks in Synthetic Aperture Radar (SAR) imagery using:

- SAR physics-based pseudo-labelling.
- A custom convolutional neural network, ProteusSegNet.

The project was developed around the GEM No.2 oil spill case study as part of the National Space Academy 2026 – Earth Observation Application Development (Level 2) course.

The Google Colab notebook remains the main application. A separate local Windows application is available in `desktop_app/`.

> All outputs indicate suspected oil, not confirmed spills. Labels are derived from SAR heuristics and ESA WorldCover reference data, not human-annotated ground truth. Independent validation has not been completed.

---

## Team & Roles

| Member | Role and contribution |
|---|---|
| Mariolino Daniel III Julabar | Project Lead & Primary Developer — SAR processing, ProteusSegNet design and application implementation |
| AlReem AlRowahi | Project design, testing and analysis support |
| Sultan AlShehhi | Project analysis support |

Advisory and feedback: Dr. Vincent Markiet, National Space Academy EO faculty, Space42.

---

## Alignment with UN SDGs & UN Water Conference 2026

Project Proteus was developed in anticipation of the UN Water Conference in Abu Dhabi in December 2026, demonstrating how Earth Observation and GeoAI can support water and marine-environment monitoring.

The project has thematic alignment with:

- SDG 14 – Life Below Water: screening for suspected marine pollution and supporting analyst review.
- SDG 6 – Clean Water and Sanitation: demonstrating satellite-based surveillance relevant to protecting water resources.
- SDG 13 – Climate Action: exploring EO tools relevant to resilient coastal monitoring.
- SDG 9 – Industry, Innovation and Infrastructure: integrating open EO data and custom models into an analytical prototype.

These are thematic connections, not measured SDG outcomes. No UN endorsement or operational certification is implied.

---

## Problem, Intended Users & Supported Decisions

### Problem

SAR can reveal oil-related suppression of sea-surface roughness as dark patches. However:

- Low-wind areas, biogenic films, currents and rain cells can resemble oil.
- Manual interpretation is time-consuming.
- Analysts need repeatable screening with transparent processing criteria.

### Intended Users

- Earth-observation analysts and marine-surveillance teams.
- Researchers and academic users studying SAR-based oil screening.
- Teams prototyping SAR and GeoAI maritime-monitoring workflows.

### Decisions Supported

Proteus helps analysts decide:

- Which candidate areas should receive visual review first.
- Which objects are larger, darker or higher-contrast.
- Which candidate features warrant further investigation using independent evidence.

Proteus is a screening and research prototype, not an operational spill-confirmation tool. It does not establish vessel responsibility, offences or environmental harm.

---

## Data & Technical Approach

### Input Data

#### 1. SAR Imagery

- Single-band, calibrated SAR backscatter in dB.
- Georeferenced GeoTIFF with a defined CRS.
- Raw and linear-backscatter imagery must be preprocessed externally.
- Suitable sources include Sentinel-1 and other calibrated SAR missions, subject to their data terms.
- Provider, acquisition date, preprocessing and licence must be retained from the source metadata.

The repository’s sample scene is identified as `SAR_20261027_Subset-2`. Consult the acquisition document for its provenance and processing details; the filename alone is not acquisition metadata.

#### 2. ESA WorldCover Reference

- Dataset: ESA WorldCover 10 m 2021 v200.
- Reference year: 2021.
- Licence: CC BY 4.0.
- DOI: https://doi.org/10.5281/zenodo.7254221
- Water class: `80`.
- Unknown coverage: `0`, excluded from analysis.

Example tile filename:

```text
ESA_WorldCover_10m_2021_v200_N24E054_Map.tif
```

WorldCover attribution:

> © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium.

The Colab workflow requires locally supplied WorldCover tiles. The updated desktop application supports local tiles or online retrieval.

WorldCover is a general land-cover reference, not a contemporaneous coastline survey. Missing coverage remains excluded.

#### 3. Model Checkpoint

ProteusSegNet uses a compatible `proteus_model.pt` checkpoint containing model weights and associated normalisation and training information.

The desktop application can train a checkpoint from a prepared dataset or load a trusted compatible checkpoint.

Only load checkpoints from trusted sources.

### Workflow

1. Validate SAR inputs: band count, CRS, pixel count and dB sanity checks.
2. Align WorldCover to the SAR grid and identify reference water.
3. Generate physics-based pseudo-labels using backscatter thresholds, local contrast and region growth.
4. Train ProteusSegNet on prepared pseudo-labels, or load a compatible checkpoint.
5. Run patch-wise model inference on normalised SAR.
6. Retain candidates satisfying the physics criteria and model-score threshold.
7. Compute object statistics and export results.

Label values:

```text
0   = evaluated background water
1   = suspected oil
255 = excluded
```

### Model & Parameters

ProteusSegNet is a custom convolutional encoder–decoder segmentation network with skip connections. Training uses physics-generated pseudo-labels rather than independently verified spill labels.

| Parameter | Default or options |
|---|---|
| Base seed threshold | −25 dB |
| Sensitivity | Conservative, Balanced, Sensitive |
| Model score threshold | 0.50 |
| Model patch size | 256 × 256 pixels |
| Minimum object size | 20 pixels |
| Local contrast window | 51 × 51 pixels |
| Zero-valued pixel exclusion | Configurable |
| Processing | Auto, Always tiled, Full scene |
| Tile size | 1024, 2048 or 4096 pixels |
| Compute device | Auto, GPU or CPU |

Sensitivity presets modify the threshold offset, minimum local contrast and region-growth allowance.

Model scores and heuristic confidence classes are not calibrated probabilities.

### Large Scenes

Automatic processing uses full-scene processing for scenes up to approximately 16 million pixels and tiled processing for larger scenes.

The tiled workflow uses overlapping processing regions, seam-aware component merging and disk-backed scratch files. Scenes up to approximately 1.5 billion pixels are permitted by the configured limit, subject to available memory and disk space.

---

## Requirements

### Main Colab Notebook

The notebook uses:

```text
numpy
torch
rasterio
scipy
pyproj
matplotlib
gradio>=5
```

Its setup cell installs missing libraries in Google Colab.

### Local Desktop Application

The desktop application uses a separate environment defined in `desktop_app/requirements.txt`:

```text
numpy==2.3.2
torch==2.9.0
rasterio>=1.4.3
scipy>=1.15.3
pyproj>=3.7.1
matplotlib>=3.10.3
Pillow>=11.2.1
affine==3.0.1
```

Some desktop dependencies specify minimum versions, so resolved environments may vary.

Do not merge the notebook and desktop requirements.

---

## Installation & How to Run

Two interfaces are available:

1. Google Colab notebook with Gradio — the main application.
2. Local Windows application — a separate local implementation.

### 1. Google Colab — Main Application

#### Requirements

- Google account.
- Modern browser.
- Colab Python 3 runtime.
- CPU runtime, or an optional GPU runtime.

Python is managed by Colab rather than installed manually.

#### Launch

Open:

https://colab.research.google.com/github/mariolinoiii/project-proteus/blob/main/ProteusSegNet.ipynb

1. Select `Runtime → Change runtime type → Python 3`.
2. Optionally select a GPU.
3. Select `Runtime → Run all`.
4. Wait for dependency setup and the Gradio interface to launch.
5. Sign in using the demo authentication configured in the notebook.

The current course-demo credentials are:

```text
Username: poseidon
Password: n5@202613v3!2
```

These credentials are public demonstration credentials, not protection for sensitive data. Do not reuse them elsewhere.

#### Usage

1. Upload calibrated, single-band SAR GeoTIFF imagery in dB.
2. Upload the required WorldCover tiles with their original filenames.
3. Supply a compatible checkpoint, unless one is already available in the runtime.
4. Choose sensitivity, backscatter threshold and model-score threshold.
5. Configure device, zero-value exclusion and processing options.
6. Click Execute analysis.
7. Review candidate objects, figures and report information.
8. Download the analysis ZIP or individual outputs.

Colab sessions are temporary. Save outputs and checkpoints before disconnecting; files may need to be uploaded again in a new runtime.

#### Expected Runtime

Runtime depends on scene size, hardware and processing settings. Small scenes may finish in minutes, while large tiled scenes can take substantially longer. GPU acceleration primarily benefits model processing.

---

### 2. Local Desktop Application — Windows

The updated Project Proteus v3 application is included under `desktop_app/`. It provides dataset preparation, physics-based pseudo-labelling, model training, scene analysis and PDF/GIS exports.

Unlike the main Colab notebook, the desktop application does not use Gradio. It runs a local Python server with an HTML, CSS and JavaScript browser interface.

#### Requirements

- Windows 10/11, 64-bit.
- Python 3.14 and the Windows `py` launcher.
- A modern browser.
- Internet for dependency installation and optional online WorldCover retrieval.
- Calibrated SAR GeoTIFFs in dB with a defined CRS.
- Optional NVIDIA GPU with a compatible CUDA-enabled PyTorch installation.

#### Setup & Launch

From the repository root:

```bat
cd desktop_app
Start_Proteus.bat
```

Alternatively, double-click `Start_Proteus.bat` inside `desktop_app/`.

The launcher creates or reuses `.venv`, installs dependencies when needed and starts the application. Keep the console window open.

Open:

```text
http://127.0.0.1:8765
```

#### Manual Setup

From the repository root:

```bat
cd desktop_app
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe application\app.py
```

Do not open `index.html` directly; the interface requires the running Python server.

#### Usage

1. Open Dataset Preparation and upload/select SAR scenes.
2. Upload WorldCover tiles or choose online retrieval.
3. Set screening parameters, prepare the dataset and inspect previews.
4. Open Model Training, select the device and epochs, then train.
5. Open Analysis Workspace and select the scene, model and settings.
6. Run analysis and review candidate objects.
7. Download the PDF report or complete analysis ZIP.

A trusted compatible checkpoint can be used instead of training a new model. Back up important checkpoints before training, which replaces the default model.

#### Configuration & Runtime

- `PROTEUS_HOME`: overrides the workspace location.
- `PROTEUS_PORT`: overrides the default port, `8765`.

Example in Command Prompt:

```bat
set "PROTEUS_HOME=D:\ProteusWorkspace"
set "PROTEUS_PORT=8766"
Start_Proteus.bat
```

Then open `http://127.0.0.1:8766`.

No fixed desktop runtime benchmark has been established. Processing depends on scene size, hardware, tile size and training epochs; large scenes may take tens of minutes or longer.

After dependency installation, processing can run without internet when all required data and model files are available locally.

---

## Sample Data & Example Results

### Sample Input

The repository includes:

```text
data/
├── ProteusSegNet_Data_Acquisition.pdf
├── ProteusSegNet_Demonstration.pdf
└── sample_input/
    ├── SAR_20261027_Subset-2.zip
    └── ESA_WorldCover_10m_2021_v200_N24E054_Map.tif
```

Consult the acquisition and demonstration documents for source details, dates, preprocessing and applicable data terms.

Extract the SAR archive before uploading its GeoTIFF.

### Example Run

The supplied example results use:

```text
Sensitivity: Conservative
Model score threshold: 0.8
```

To compare a new run with the example, use the same imagery, WorldCover tile, compatible checkpoint and complete processing settings. Matching only the sensitivity and score threshold does not guarantee identical results.

### Output Contents

Typical outputs include:

```text
suspected_oil_labels.tif
model_scores.tif
oil_object_ids.tif
oil_objects.geojson
oil_objects.csv
report.json
oil_preview.png
backscatter_histogram.png
object_statistics.png
```

The updated desktop application also exports a PDF report and GIS-ready products, including an RGB overlay and scene footprint.

Example desktop downloads:

```text
ProteusSegNet_Analysis_Report_SAR_20261027_Subset-2.pdf
ProteusSegNet_Analysis_Results_SAR_20261027_Subset-2.zip
ProteusSegNet_Dataset.zip
```

Existing example outputs are not automatically regenerated by updating the desktop application.

---

## Results & Limitations

The application produces candidate dark-feature maps, object measurements and heuristic review-priority information. These demonstrate the workflow, not independently established spill-detection accuracy.

Key limitations:

- Low wind, biogenic films, rain and other look-alikes can be selected.
- Thin or low-contrast oil may be missed.
- Training labels are physics-generated pseudo-labels, not ground truth.
- Training loss measures fit to those labels, not independent accuracy.
- Model scores and confidence classes are not calibrated probabilities.
- WorldCover age and coverage affect the evaluation area.
- Area estimates depend on CRS geometry and pixel size.
- Large scenes require sufficient RAM and scratch-disk space.
- Connected candidate objects are not necessarily separate spills.

Outputs require experienced analyst review and independent evidence. Do not use them alone for spill confirmation, enforcement, attribution or environmental-harm assessment.

---

## Repository Structure

```text
project-proteus/
├── ProteusSegNet.ipynb                     # Main Google Colab + Gradio application
├── requirements.txt                       # Notebook environment requirements
├── data/
│   ├── ProteusSegNet_Data_Acquisition.pdf  # Data acquisition document
│   ├── ProteusSegNet_Demonstration.pdf     # Demonstration case description
│   └── sample_input/
│       ├── SAR_20261027_Subset-2.zip
│       └── ESA_WorldCover_10m_2021_v200_N24E054_Map.tif
├── results/                               # Example analysis outputs
│   ├── backscatter_histogram.png
│   ├── model_scores.tif
│   ├── object_statistics.png
│   ├── oil_object_ids.tif
│   ├── oil_objects.csv
│   ├── oil_objects.geojson
│   ├── oil_preview.png
│   ├── report.json
│   └── suspected_oil_labels.tif
├── desktop_app/
│   ├── Start_Proteus.bat                   # Windows launcher
│   ├── bootstrap.py                       # Dependency setup
│   ├── requirements.txt                   # Desktop-specific dependencies
│   ├── application/
│   │   ├── app.py                         # Local server and entry point
│   │   ├── services.py                    # Jobs and workspace services
│   │   ├── proteus_core.py                # SAR processing, model and analysis
│   │   ├── previews.py                    # Dataset previews and charts
│   │   ├── report_pdf.py                  # PDF report generation
│   │   └── web/
│   │       ├── index.html                 # Local browser interface
│   │       ├── app.js                      # Interface behaviour
│   │       └── style.css                  # Interface styling
│   └── workspace/                         # Generated locally during use
└── README.md
```

The notebook and desktop application use separate environments. Generated `.venv/` and `workspace/` folders should remain local and should not be committed.

---

## Methodology & Validation Status

The processing sequence is:

```text
Input checks
→ WorldCover alignment
→ Physics-based pseudo-labelling
→ ProteusSegNet inference
→ Physics/model fusion
→ Object analysis
→ Export
```

Current validation status:

- No dedicated, independently human-annotated test set has been established.
- No independently measured IoU, precision or recall is reported.
- Current use is exploratory and educational.

Planned work:

- Create a manually labelled GEM No.2 validation subset.
- Compare physics-only screening with physics-plus-ProteusSegNet screening.
- Evaluate generalisation to other spill and non-spill SAR scenes.

---

## Security & Data Handling

- The Colab Gradio interface may use a public sharing link. Treat its demo credentials as public.
- Do not upload sensitive imagery to a shared demonstration session.
- Do not commit personal credentials, API keys or private data.
- The desktop application runs locally; do not expose its server publicly without a separate security review.
- Online WorldCover retrieval and dependency installation require external network requests.
- Desktop uploads, models and outputs remain in the local workspace until removed.
- Only load trusted model checkpoints.
- Sample data and third-party software remain subject to their respective licences.

---

## AI-Assisted Development

AI tools, including ChatGPT, Claude, Gemini, gIQ AI Assistant and Perplexity, supported:

- Code generation and refactoring.
- Debugging and performance-related development.
- Documentation and usage guidance.

Architectural choices, final code, testing and interpretation remain the responsibility of Team Proteus.
