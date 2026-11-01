# Project Proteus

**Project Proteus** maps suspected marine oil slicks using Synthetic Aperture Radar (SAR) imagery, local ESA WorldCover reference data, and a custom ProteusSegNet deep-learning model.

The project was developed as part of the National Space Academy 2026 Earth Observation Application Development Course.

## Project overview

Project Proteus uses a hybrid workflow that combines:

- Calibrated, single-band SAR imagery in dB
- Local ESA WorldCover water-reference data
- SAR backscatter and local-contrast analysis
- A custom convolutional encoder-decoder model, ProteusSegNet
- Physics-based filtering to reduce false detections

The application produces suspected-oil masks, object-level statistics, analytical figures, and downloadable result packages.

## Project versions

This repository contains two versions of Project Proteus:

| Version | File | Purpose |
|---|---|---|
| Google Colab notebook | `ProteusSegNet.ipynb` | Interactive browser-based application |
| Python desktop application | `proteus_app.py` | Local Windows/Mac desktop application |

## Main Colab notebook

The main Google Colab notebook is:

[ProteusSegNet.ipynb](ProteusSegNet.ipynb)

### How to run the Colab notebook

1. Open `ProteusSegNet.ipynb` in Google Colab.
2. Connect to a runtime.
3. Run the notebook cells from top to bottom.
4. Upload the required SAR GeoTIFF and matching local ESA WorldCover GeoTIFF files.
5. Upload a trained model checkpoint, if available.
6. Run the analysis and download the generated results package.

## Python desktop application

The desktop application can be run locally using Python.

### How to run the desktop application

1. Install Python 3.10 or newer.
2. Open a terminal or command prompt in the project folder.
3. Install the desktop requirements:

```bash
pip install -r requirements-desktop.txt
```

4. Run the application:

```bash
python proteus_app.py
```

5. Open the local Gradio link shown in the terminal.

## Input data

Both versions accept:

- A calibrated, georeferenced, single-band SAR GeoTIFF expressed in dB.
- Matching local ESA WorldCover 2021 GeoTIFF reference tiles.
- An optional trained ProteusSegNet checkpoint saved as `proteusmodel.pt`.

The maximum supported SAR input size is 16 million pixels.

## Outputs

After processing, the application can generate:

- Suspected-oil label masks
- Model score rasters
- Oil object IDs
- Object-level CSV statistics
- GeoJSON polygons
- Analytical figures
- A downloadable ZIP results package

## Requirements

This repository has two requirements files:

| File | Used for |
|---|---|
| `requirements.txt` | Google Colab notebook |
| `requirements-desktop.txt` | Local Python desktop application |

### Colab notebook requirements

`requirements.txt` contains:

```text
numpy
torch
rasterio
scipy
pyproj
matplotlib
gradio==4.44.7
```

### Desktop application requirements

`requirements-desktop.txt` contains:

```text
numpy>=1.26,<3
torch>=2.6,<3
rasterio>=1.3,<2
scipy>=1.11,<2
pyproj>=3.6,<4
matplotlib>=3.8,<4
gradio>=5,<6
```

The Colab notebook uses Gradio 4.44.7, while the desktop application uses Gradio 5.

## Important notes

- The application identifies **suspected oil**, not confirmed oil spills.
- Model scores and confidence categories are not calibrated probabilities.
- Independent validation has not yet been completed.
- Do not publish the notebook publicly without removing any authentication credentials.

## Author

**Mariolino Daniel III Julabar**  
**AlReem Saeed**  
**Sultan AlShehhi**  
