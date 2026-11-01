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

## Main notebook

The main Google Colab notebook is:

[ProteusSegNet.ipynb](ProteusSegNet.ipynb)

## How to run

1. Open `ProteusSegNet.ipynb` in Google Colab.
2. Connect to a runtime.
3. Run the notebook cells from top to bottom.
4. Upload the required SAR GeoTIFF and matching local ESA WorldCover GeoTIFF files.
5. Upload a trained model checkpoint, if available.
6. Run the analysis and download the generated results package.

## Input data

The notebook accepts:

- A calibrated, georeferenced, single-band SAR GeoTIFF expressed in dB.
- Matching local ESA WorldCover 2021 GeoTIFF reference tiles.
- An optional trained ProteusSegNet checkpoint saved as `proteusmodel.pt`.

The maximum supported SAR input size is 16 million pixels.

## Outputs

After processing, the notebook can generate:

- Suspected-oil label masks
- Model score rasters
- Oil object IDs
- Object-level CSV statistics
- GeoJSON polygons
- Analytical figures
- A downloadable ZIP results package

## Requirements

The required Python packages are listed in [`requirements.txt`](requirements.txt):

```text
numpy
torch
rasterio
scipy
pyproj
matplotlib
gradio==4.44.7
```

## Important notes

- The application identifies **suspected oil**, not confirmed oil spills.
- Model scores and confidence categories are not calibrated probabilities.
- Independent validation has not yet been completed.
- Do not publish the notebook publicly without removing any authentication credentials.

## Author

**Mariolino Daniel III Julabar**  
**AlReem Saeed**  
**Sultan AlShehhi**  
