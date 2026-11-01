# Project Proteus — Google Colab Notebook

Mapping suspected oil slicks using Synthetic Aperture Radar (SAR) imagery, local ESA WorldCover reference data, and ProteusSegNet.

This README describes `ProteusSegNet.ipynb` only. The local Python application is separate and is located in the `desktop_app/` folder.

## Overview

Project Proteus is a research and educational prototype for mapping suspected oil slicks.

The notebook combines:

- Local WorldCover water-reference masking
- SAR backscatter thresholds and local-contrast analysis
- A custom ProteusSegNet convolutional encoder-decoder model
- Physics-based filtering of model predictions
- Object-level statistics, maps, and downloadable results

Outputs indicate suspected oil, not confirmed oil spills.

## Main notebook

[View ProteusSegNet.ipynb](ProteusSegNet.ipynb)

[Open in Google Colab](https://colab.research.google.com/github/mariolinoiii/project-proteus/blob/main/ProteusSegNet.ipynb)

If the repository is private, you must have repository access and authorize Colab to access private GitHub repositories.

## What you need

For analysis, prepare:

1. A calibrated, georeferenced, single-band SAR GeoTIFF expressed in dB.
2. The matching local ESA WorldCover 2021 reference GeoTIFF files.
3. A compatible trained ProteusSegNet checkpoint, normally named `proteusmodel.pt`.

Keep the original WorldCover filenames unchanged.

Model weights and input imagery are not embedded in the notebook.

## How to run an analysis

1. Open `ProteusSegNet.ipynb` in Google Colab.
2. Connect to a runtime.
3. Select **Runtime → Run all**.
4. Wait for the application interface to appear.
5. Sign in using the application credentials provided separately by the project maintainer.
6. Open the **Analysis Workspace** tab.
7. Upload your SAR GeoTIFF.
8. Upload the matching WorldCover reference files.
9. Upload your compatible trained model checkpoint.
10. Review the processing settings.
11. Select **Execute analysis**.
12. Review the report and analytical figures.
13. Download the analysis ZIP using the control beneath the figures.

CPU is the default processing option.

In a fresh Colab runtime, upload the model checkpoint again. If the default checkpoint already exists in the current runtime, the model upload field may be left empty.

## Optional dataset preparation and training

Training is not required when you already have a compatible checkpoint.

To train a new model:

1. Open **Dataset Preparation**.
2. Upload the training SAR images and matching WorldCover reference files.
3. Select **Prepare annotated dataset**.
4. Review the dataset preparation report.
5. Open **Model Training**.
6. Set the number of training epochs.
7. Confirm that you authorize training and replacement of the default checkpoint.
8. Select **Start model training**.
9. Download the resulting `proteusmodel.pt` checkpoint.

Save any existing checkpoint before training: the default checkpoint may be overwritten.

The training labels are generated using SAR physics rules. They are pseudo-labels, not independently verified oil-spill annotations.

## Analysis outputs

The analysis ZIP can contain:

- Suspected-oil label raster
- Model score raster
- Oil-object ID raster
- Object statistics in CSV format
- Object polygons in GeoJSON format
- Processing report in JSON format
- SAR and suspected-oil preview maps
- Backscatter distribution histogram
- Object area and contrast figure

The trained model checkpoint is downloaded separately from the **Model Training** tab, not from the analysis ZIP.

## Notebook requirements

The root-level `requirements.txt` is for this notebook:

```text
numpy
torch
rasterio
scipy
pyproj
matplotlib
gradio==4.44.7
```

The notebook includes a setup cell that installs missing libraries.

The desktop application has its own requirements file:

```text
desktop_app/requirements.txt
```

Do not combine the notebook and desktop requirements into one installation environment.

## Limitations

- Maximum SAR input size: 16,000,000 pixels per image.
- Automatic splitting of oversized scenes and merging of outputs are not implemented.
- Independent validation has not been completed.
- Training-image results are demonstrations, not independent accuracy tests.
- Model scores are not calibrated probabilities of oil.
- Object confidence categories are heuristic.
- WorldCover water classes are not specifically marine-water classes.
- Missing reference coverage remains excluded.
- Area estimates depend on coordinate-system geometry and are not confirmed spill measurements.
- Processing and large embedded downloads may require substantial runtime and browser memory.

ProteusSegNet is a custom convolutional encoder-decoder with skip connections. It is not an implementation of the published SegNet architecture.

## Security

Do not publish passwords, API keys, or other credentials in notebook code, saved outputs, or documentation.

Keep authentication credentials outside the committed notebook and share them separately with authorized users.

## Separate desktop application

The local Python application and its Windows setup scripts are located in:

[desktop_app/](desktop_app/)

Use the instructions and requirements inside that folder for the desktop version. This README covers the Colab notebook only.

## Developer

**Mariolino Daniel III Julabar**  
**AlReem Saeed**  
**Sultan AlShehhi**  
