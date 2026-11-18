# Project Proteus v3

**Mapping of Suspected Oil Slicks using SAR & GeoAI**

ProteusSegNet is a local research application for screening calibrated SAR imagery for suspected oil features. It combines dataset preparation, physics-based pseudo-labelling, model training and analyst-review exports in one workflow.

**Suspected oil only. Not a confirmed spill. Independent validation has not been completed.**

## Team

| Member | Role | Contribution |
|---|---|---|
| Mariolino Daniel III Julabar | Project Lead & Primary Developer | SAR processing, network architecture and application implementation |
| AlReem AlRowahi | Project design, testing and analytical support | Design, Testing & Analysis Support |
| Sultan AlShehhi | Project analysis support | Analysis Support |

**Faculty Advisor:** Dr. Vincent Markiet, Space42.

AI tools supported code generation, refactoring, debugging and documentation. Team Proteus remains responsible for architectural choices, final code and interpretation.

## Problem and intended users

Dark features in SAR imagery can be caused by oil, but also by low wind, biogenic films, rain cells and other look-alikes. This application helps users identify and prioritise candidate dark features for visual review.

It is intended for Earth-observation analysts, marine-surveillance researchers and academic users who need a structured starting point for reviewing possible oil-slick candidates.

The application supports a screening decision: **which areas or objects should an analyst review first?** It does **not** confirm spills, attribute vessels, establish offences, quantify environmental harm, or replace independent expert review.

## Data and technical approach

### Input imagery

The application accepts a calibrated, single-band SAR backscatter GeoTIFF in dB with a defined coordinate reference system (CRS).

- Input must be preprocessed before upload.
- Raw, linear-backscatter or multi-band imagery is not accepted.
- The user must retain the source mission’s attribution, acquisition information and licence terms.
- No sample imagery or pretrained model checkpoint is bundled with this repository.

### Reference data

The application uses ESA WorldCover 2021 v200 as a land/water reference.

- **Dataset:** ESA WorldCover 2021 v200
- **Licence:** CC BY 4.0
- **DOI:** https://doi.org/10.5281/zenodo.7254221
- **Attribution:** © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium.
- Sentinel data are subject to the Copernicus Sentinel legal notice; other missions retain their own licences.

WorldCover is used to identify reference water. Missing or unknown reference coverage remains excluded from analysis. WorldCover 2021 is not a contemporaneous coastline survey.

### Workflow

1. Upload a calibrated, single-band SAR GeoTIFF in dB with a CRS.
2. Select local WorldCover reference tiles or use online retrieval.
3. Generate physics-based suspected-oil pseudo-labels.
4. Inspect previews, reference coverage and label statistics.
5. Train ProteusSegNet on the prepared dataset.
6. Run scene analysis using SAR physics criteria and the trained model score threshold.
7. Review candidate objects, figures and limitations.
8. Export the PDF report and GIS-ready analysis package.

### Model and parameters

ProteusSegNet is a custom convolutional encoder-decoder segmentation network with skip connections. It is trained from random weights on physics-generated pseudo-labels, not on independently verified spill labels.

Default screening parameters include:

- Base seed threshold: -25 dB
- Model score threshold: 0.50
- Sensitivity modes: Conservative, Balanced and Sensitive
- Zero-valued pixel exclusion: enabled by default
- Processing: Auto, Always tiled, or Full scene
- Tile size: 1024, 2048 or 4096 pixels
- Compute device: Auto, GPU or CPU

Training loss measures how well the model fits the physics-generated pseudo-labels. It is **not** independent detection accuracy.

## Installation

### Requirements

- Windows 10 or 11, 64-bit
- Python 3.10 or newer; Python 3.12 is recommended
- Windows `py` launcher
- A modern web browser
- Internet access for first-time dependency installation
- Optional: a working NVIDIA GPU and compatible CUDA-enabled PyTorch installation

Install Python yourself from https://www.python.org/downloads/windows/ if it is not already installed. The application does **not** install Python or graphics drivers automatically.

### Recommended launch method

1. Extract the entire ZIP archive to a writable folder.
2. Double-click `Start_Proteus.bat`.
3. Keep the console window open.
4. Open the application at: http://127.0.0.1:8765

The launcher creates or reuses a local `.venv` virtual environment, installs dependencies when needed, and starts the application.

### Dependency versions

`requirements.txt` uses pinned package versions to make the application environment more reproducible:

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

PyTorch is listed at version 2.7.1, but it must be installed from the correct package index for the intended device:

- **CPU-only installation:**

```bat
.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

- **CUDA GPU installation:** install the appropriate CUDA-enabled PyTorch 2.7.1 build manually using the official PyTorch instructions for your CUDA and NVIDIA driver version.

The launcher and `bootstrap.py` install the remaining pinned dependencies from `requirements.txt`. They do not automatically install Python, NVIDIA drivers, or a GPU-specific PyTorch build.

### Manual setup

From the extracted project root, run:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe application\app.py
```

For a CPU-only setup, the command above installs the CPU build of PyTorch and then installs the remaining pinned dependencies.

If you use a GPU, install the correct CUDA-enabled PyTorch 2.7.1 build manually before running:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe application\app.py
```

## How to use the app

1. Open **Dataset Preparation**.
2. Upload a calibrated, single-band SAR GeoTIFF in dB with a CRS.
3. Select one or more SAR scenes.
4. Choose **Local upload** or **Online** for WorldCover reference coverage.
5. Select the sensitivity mode and base seed threshold.
6. Click **Prepare dataset**.
7. Review the scene preview, WorldCover reference, pseudo-label overlay and label graphs.
8. Open **Model Training**.
9. Select the compute device and number of epochs.
10. Click **Start training**.
11. Open **Analysis Workspace**.
12. Select the SAR scene, reference option, model and analysis settings.
13. Click **Run analysis**.
14. Review the candidate objects, figures, screening summary and limitations.
15. Download the PDF report and/or complete analysis ZIP.

## Example input and output

### Example input

```text
SAR_20261021_Subset-2.tif
```

This filename is an example only. It represents a calibrated, single-band SAR backscatter GeoTIFF in dB with a CRS.

### Example outputs

```text
ProteusSegNet_Analysis_Report_SAR_20261021_Subset-2.pdf
ProteusSegNet_Analysis_Results_SAR_20261021_Subset-2.zip
ProteusSegNet_Dataset.zip
```

These are naming examples, not claimed detections.

The analysis ZIP may include:

- Suspected-oil label raster in the SAR scene’s native CRS
- Model-score raster
- Object-ID raster
- RGB overlay raster
- WGS84 GeoJSON object polygons
- WGS84 GeoJSON scene footprint
- Object CSV table
- JSON metadata and report
- Analysis figures
- PDF report

## Results and limitations

The application produces a **suspected-oil screening product** for analyst review.

Results depend on:

- SAR calibration and dB units
- Image CRS and geometry
- WorldCover reference age and coverage
- Screening thresholds and sensitivity mode
- Model training data and score threshold
- Presence of look-alikes and low-contrast oil

Important limitations:

- Low wind, biogenic films and weather can resemble oil.
- Low-contrast oil may be missed.
- Areas depend on CRS geometry and reference coverage.
- Model scores and confidence classes are heuristic, not calibrated probabilities.
- Screening levels and review flags are not regulatory thresholds.
- Training loss is not independent detection accuracy.
- No independent precision, recall, IoU or spill-confirmation validation is claimed.
- The application does not provide vessel attribution or environmental-harm quantification.

Do not use this application alone for spill confirmation, enforcement, attribution or environmental-harm assessment.

## Configuration

The following environment variables can be used:

```text
PROTEUS_HOME
```

Overrides the default workspace folder.

```text
PROTEUS_PORT
```

Overrides the default application port, 8765.

Changing these settings does not automatically migrate existing uploads, models, datasets or outputs.

## Troubleshooting

- **Invalid SAR:** check that the file is single-band, calibrated backscatter in dB, and has a CRS.
- **No trained model:** prepare a dataset and train a model, or select a trusted compatible checkpoint.
- **Missing WorldCover reference:** upload the required tiles or choose Online.
- **CUDA failure:** select Auto or CPU; manually check the PyTorch build and NVIDIA driver if GPU use is required.
- **Disk-space error:** free storage or move the workspace and scratch folder to a larger drive.
- **Connection failure:** keep the console open, check the error message and reopen the local URL.
- **Port already in use:** stop the previous Proteus instance or use a different port.

## Updating an existing installation

Stop the old application and back up the working installation before updating.

Replace the application folder and root launcher, setup and documentation files. Preserve the existing `.venv` and workspace folders if needed.

Existing PDF reports and dataset previews are not automatically reformatted. Run a new analysis or prepare a new dataset to generate updated outputs.

## Licence and attribution

Third-party data and software licences continue to apply.

- ESA WorldCover 2021 v200: CC BY 4.0
- Copernicus Sentinel data: subject to the Copernicus Sentinel legal notice