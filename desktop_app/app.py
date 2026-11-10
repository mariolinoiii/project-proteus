"""Project Proteus / ProteusSegNet — Windows local application.
Run with Start_Proteus.bat after Setup_Windows.bat.
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# 1. IMPORTS AND CONFIGURATION
# ==================================================

import os
import csv
import json
import shutil

import numpy as np
import rasterio

from scipy import ndimage as ndi
from pyproj import CRS
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds


# Local WorldCover GeoTIFFs go in this folder.
# This code NEVER downloads them.
WORLD_COVER_FOLDER = str(BASE_DIR / "worldcover_cache")

OPERATIONAL_MODES = {
    "Conservative": {
        "threshold_offset_db": -2.0,
        "contrast_min": 0.15,
        "growth_allowance_db": 1.0,
    },
    "Balanced": {
        "threshold_offset_db": 0.0,
        "contrast_min": 0.12,
        "growth_allowance_db": 2.0,
    },
    "Sensitive": {
        "threshold_offset_db": 2.0,
        "contrast_min": 0.08,
        "growth_allowance_db": 3.0,
    },
}

os.makedirs(WORLD_COVER_FOLDER, exist_ok=True)


import io
import contextlib
import tempfile
import zipfile
import base64
import sys
import html
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr


# ==================================================
# 2. LOCAL REFERENCE AND SAR PHYSICS FUNCTIONS
# ==================================================

def get_operational_params(mode, base_threshold_db):
    """Return the threshold, minimum contrast, and growth allowance."""
    if mode not in OPERATIONAL_MODES:
        raise ValueError(
            f"Unknown mode: {mode}. Choose from {list(OPERATIONAL_MODES)}"
        )

    preset = OPERATIONAL_MODES[mode]

    return (
        float(base_threshold_db) + preset["threshold_offset_db"],
        preset["contrast_min"],
        preset["growth_allowance_db"],
    )


def worldcover_tile_name(lat, lon):
    """Construct the filename coordinate component used by WorldCover."""
    north_south = "N" if lat >= 0 else "S"
    east_west = "E" if lon >= 0 else "W"

    return (
        f"{north_south}{abs(lat):02d}"
        f"{east_west}{abs(lon):03d}"
    )


def load_local_water_reference(src):
    """
    Align local WorldCover tiles to the SAR image.

    Returns:
        reference: uint8 array.
            0 = unknown
            1 = non-water
            2 = water
        sources: list of local reference files used.

    No network requests are made.
    """
    if src.crs is None:
        raise ValueError("The SAR image must have a CRS.")

    west, south, east, north = transform_bounds(
        src.crs,
        "EPSG:4326",
        *src.bounds,
        densify_pts=21,
    )

    if (
        not (-180 <= west < east <= 180)
        or not (-90 <= south < north <= 90)
        or east - west > 12
        or north - south > 12
    ):
        raise ValueError(
            "Use a regional SAR image no more than 12 degrees across "
            "and without antimeridian crossing."
        )

    reference = np.zeros(
        (src.height, src.width),
        dtype=np.uint8,
    )

    known_classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    sources = []
    missing = []

    latitudes = range(
        int(np.floor(south / 3) * 3),
        int(np.floor((north - 1e-9) / 3) * 3) + 1,
        3,
    )

    longitudes = range(
        int(np.floor(west / 3) * 3),
        int(np.floor((east - 1e-9) / 3) * 3) + 1,
        3,
    )

    for lat in latitudes:
        for lon in longitudes:
            tile_id = worldcover_tile_name(lat, lon)

            filename = (
                f"ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
            )
            path = os.path.join(WORLD_COVER_FOLDER, filename)

            if not os.path.isfile(path):
                missing.append(filename)
                continue

            with rasterio.open(path) as wc:
                with WarpedVRT(
                    wc,
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    src_nodata=0,
                    nodata=0,
                    resampling=Resampling.nearest,
                ) as aligned:
                    for row in range(0, src.height, 512):
                        height = min(512, src.height - row)

                        window = rasterio.windows.Window(
                            0, row, src.width, height
                        )
                        values = aligned.read(1, window=window)
                        target = reference[row:row + height]

                        target[
                            np.isin(values, known_classes)
                            & (values != 80)
                        ] = 1

                        target[values == 80] = 2

            sources.append(path)

    if not sources:
        raise FileNotFoundError(
            "Required local WorldCover files were not found.\n"
            f"Folder: {WORLD_COVER_FOLDER}\n"
            "Expected files:\n"
            + "\n".join(missing)
        )

    if missing:
        print(
            "  Warning: some reference tiles are missing. "
            "Their pixels will remain excluded."
        )

    return reference, sources


def calculate_local_contrast(data, water, window_size=51):
    """Measure SAR darkness relative to nearby valid water."""
    # Invalid/excluded pixels contribute neither intensity nor sample count.
    linear = np.zeros(data.shape, dtype=np.float64)
    linear[water] = np.power(10.0, data[water] / 10.0)

    if not np.isfinite(linear[water]).all():
        raise ValueError(
            "SAR values overflowed during conversion from dB. "
            "Check that the input really contains calibrated backscatter in dB."
        )

    local_sum = ndi.uniform_filter(
        linear,
        size=window_size,
        mode="reflect",
    )
    local_count = ndi.uniform_filter(
        water.astype(np.float64),
        size=window_size,
        mode="reflect",
    )

    local_mean = np.zeros_like(linear)
    np.divide(
        local_sum,
        local_count,
        out=local_mean,
        where=local_count > 0,
    )

    contrast = np.zeros_like(linear)
    denominator = local_mean + linear

    np.divide(
        local_mean - linear,
        denominator,
        out=contrast,
        where=water & (denominator > 0),
    )

    return np.clip(contrast, -1.0, 1.0)


def pixel_area_by_row(src):
    """Return pixel areas in square metres, or None if unsupported."""
    crs = CRS.from_user_input(src.crs)
    transform = src.transform

    if crs.is_projected and len(crs.axis_info) >= 2:
        area = (
            abs(
                transform.a * transform.e
                - transform.b * transform.d
            )
            * crs.axis_info[0].unit_conversion_factor
            * crs.axis_info[1].unit_conversion_factor
        )
        return np.full(src.height, area, dtype=np.float64)

    if crs.is_geographic:
        if (
            abs(transform.b) > 1e-12
            or abs(transform.d) > 1e-12
            or len(crs.axis_info) < 2
            or any(
                abs(axis.unit_conversion_factor - np.pi / 180) > 1e-10
                for axis in crs.axis_info[:2]
            )
        ):
            return None

        geod = crs.get_geod()
        areas = np.zeros(src.height, dtype=np.float64)

        for row in range(src.height):
            y0 = transform.f + row * transform.e
            y1 = y0 + transform.e
            x0 = transform.c
            x1 = x0 + transform.a

            area, _ = geod.polygon_area_perimeter(
                [x0, x1, x1, x0],
                [y0, y0, y1, y1],
            )
            areas[row] = abs(area)

        return areas

    return None


def create_physics_labels(
    src,
    threshold_db,
    contrast_min,
    growth_allowance_db,
    zero_exclusion=True,
    min_pixels=20,
):
    """
    Generate suspected-oil pseudo-labels without a pretrained model.

    Returns:
        labels: 0 = evaluated non-oil, 1 = suspected oil, 255 = excluded.
        report: area estimates, object counts, and configuration.
    """
    if src.count != 1:
        raise ValueError("The SAR input must contain exactly one band.")

    # Full-scene processing needs memory. Use SAR tiles rather than huge scenes.
    if src.width * src.height > 16_000_000:
        raise ValueError(
            "This release accepts up to 16 million pixels "
            "per input. Split larger scenes into smaller georeferenced tiles."
        )

    data = src.read(1, out_dtype="float32")
    valid = (src.read_masks(1) > 0) & np.isfinite(data)

    if zero_exclusion:
        valid &= data != 0

    reference, reference_sources = load_local_water_reference(src)
    water = valid & (reference == 2)

    if not water.any():
        raise ValueError("No valid reference-water pixels were found.")

    contrast = calculate_local_contrast(data, water)

    seeds = (
        water
        & (data < threshold_db)
        & (contrast > contrast_min)
    )

    allowed = (
        water
        & (data < threshold_db + growth_allowance_db)
    )

    grown = ndi.binary_propagation(
        seeds,
        structure=np.ones((3, 3), dtype=bool),
        mask=allowed,
    )

    # Remove small connected objects.
    components, _ = ndi.label(grown)
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_pixels
    keep[0] = False

    oil = keep[components]

    labels = np.full(data.shape, 255, dtype=np.uint8)
    labels[water] = 0
    labels[oil] = 1

    object_ids, object_count = ndi.label(oil)
    areas_by_row = pixel_area_by_row(src)

    oil_area = None
    object_areas = None

    if areas_by_row is not None:
        oil_area = float(
            np.dot(oil.sum(axis=1), areas_by_row) / 1_000_000
        )

        object_areas = np.zeros(object_count + 1, dtype=np.float64)
        for row in range(src.height):
            object_areas += (
                np.bincount(
                    object_ids[row],
                    minlength=object_count + 1,
                )
                * areas_by_row[row]
                / 1_000_000
            )

    background = water & ~oil
    mean_background = (
        float(data[background].mean())
        if background.any()
        else None
    )

    counts = {
        "high": 0,
        "medium": 0,
        "low": 0,
        "unknown": 0,
    }

    if object_count:
        object_means = np.atleast_1d(
            ndi.mean(
                data,
                labels=object_ids,
                index=np.arange(1, object_count + 1),
            )
        )

        for object_id, object_mean in enumerate(object_means, start=1):
            area = (
                float(object_areas[object_id])
                if object_areas is not None
                else None
            )

            if mean_background is None:
                confidence = "unknown"
            else:
                contrast_db = mean_background - float(object_mean)

                if contrast_db >= 4.0 and (area is None or area >= 0.1):
                    confidence = "high"
                elif contrast_db >= 2.0 and (area is None or area >= 0.02):
                    confidence = "medium"
                else:
                    confidence = "low"

            counts[confidence] += 1

    report = {
        "model": "Local SAR physics pseudo-labeler",
        "estimated_oil_area_km2": oil_area,
        "selected_oil_pixels": int(oil.sum()),
        "evaluated_pixels": int(water.sum()),
        "excluded_pixels": int((~water).sum()),
        "threshold_db": float(threshold_db),
        "contrast_min": float(contrast_min),
        "growth_allowance_db": float(growth_allowance_db),
        "zero_exclusion": bool(zero_exclusion),
        "reference_sources": reference_sources,
        "object_summary": {
            "num_objects": int(object_count),
            "num_high_confidence": counts["high"],
            "num_medium_confidence": counts["medium"],
            "num_low_confidence": counts["low"],
            "num_unknown_confidence": counts["unknown"],
        },
    }

    return labels, report




class ProteusSegNet(nn.Module):
    """Small encoder-decoder with skip connections and one output logit."""

    def __init__(self):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(1, 16)
        self.enc2 = block(16, 32)
        self.middle = block(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.dec2 = block(64 + 32, 32)
        self.dec1 = block(32 + 16, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        first = self.enc1(x)
        second = self.enc2(self.pool(first))
        middle = self.middle(self.pool(second))

        up = nn.functional.interpolate(
            middle,
            size=second.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up = self.dec2(torch.cat([up, second], dim=1))

        up = nn.functional.interpolate(
            up,
            size=first.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up = self.dec1(torch.cat([up, first], dim=1))

        return self.output(up)

# ==================================================
# 3. BATCH DATASET GENERATION
# ==================================================

def save_binary_geotiff(path, array, src):
    """Save a uint8 mask on exactly the same grid as the SAR input."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=src.height,
        width=src.width,
        count=1,
        dtype="uint8",
        crs=src.crs,
        transform=src.transform,
        compress="deflate",
        BIGTIFF="IF_SAFER",
        nodata=None,
    ) as dst:
        dst.write(array.astype(np.uint8), 1)


def batch_generate_oil_labels(
    input_dir: str,
    output_dir: str,
    device_choice: str = "Auto",
    mode: str = "Balanced",
    base_threshold_db: float = -25.0,
    zero_exclusion: bool = True,
    quick_preview: bool = False,
    processing_strategy: str = "Auto",
    tile_size: int = 2048,
    progress=None,
):
    """
    Generate a training dataset from local single-band SAR GeoTIFFs in dB.

    This standalone version uses SAR physics, not GroundedSAM.
    Processing runs on CPU; device_choice is retained for familiar usage.

    Returns:
        dict: dataset index path, successful-image count, and failure records.
    """
    if quick_preview:
        raise ValueError(
            "Use quick_preview=False when generating training labels."
        )

    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    if os.path.commonpath([input_dir, output_dir]) == output_dir:
        raise ValueError("The input folder must not be inside the output folder.")

    files = sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.lower().endswith((".tif", ".tiff"))
        and os.path.isfile(os.path.join(input_dir, filename))
    )

    if not files:
        raise FileNotFoundError(
            f"No SAR .tif or .tiff files found in {input_dir}"
        )

    image_ids = [
        os.path.splitext(os.path.basename(path))[0]
        for path in files
    ]

    if len(set(name.casefold() for name in image_ids)) != len(image_ids):
        raise ValueError(
            "Two inputs have the same filename stem. "
            "Rename them so every image has a unique name."
        )

    # Refuse to overwrite an existing dataset.
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            "The output folder is not empty. "
            "Choose a new output_folder to avoid overwriting your dataset."
        )

    folders = {
        name: os.path.join(output_dir, name)
        for name in ["images", "masks", "validity", "reports"]
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    threshold, contrast, growth = get_operational_params(
        mode, base_threshold_db
    )

    columns = [
        "image_id",
        "sar_path",
        "label_path",
        "estimated_oil_area_km2",
        "num_objects",
        "num_high_confidence",
        "num_medium_confidence",
        "num_low_confidence",
        "validity_path",
        "report_path",
    ]

    index_path = os.path.join(output_dir, "dataset_index.csv")
    failures = []
    success_count = 0

    with open(index_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        handle.flush()

        for number, (source_path, image_id) in enumerate(
            zip(files, image_ids), start=1
        ):
            print(
                f"Processing [{number}/{len(files)}]: "
                f"{os.path.basename(source_path)}"
            )

            extension = os.path.splitext(source_path)[1].lower()

            image_path = os.path.join(
                folders["images"], image_id + extension
            )
            label_path = os.path.join(
                folders["masks"], image_id + "_label.tif"
            )
            validity_path = os.path.join(
                folders["validity"], image_id + "_valid.tif"
            )
            report_path = os.path.join(
                folders["reports"], image_id + ".json"
            )

            created_paths = []

            try:
                with rasterio.open(source_path) as src:
                    if should_use_tiling(src.width, src.height, processing_strategy):
                        print(
                            f"  Large scene ({src.width * src.height:,} px): "
                            "tiled processing"
                        )
                        created_paths += [label_path, validity_path]
                        report = tiled_physics_labels_to_files(
                            src,
                            threshold_db=threshold,
                            contrast_min=contrast,
                            growth_allowance_db=growth,
                            zero_exclusion=zero_exclusion,
                            label_path=label_path,
                            validity_path=validity_path,
                            tile_size=tile_size,
                            progress=progress,
                        )
                    else:
                        labels, report = create_physics_labels(
                            src=src,
                            threshold_db=threshold,
                            contrast_min=contrast,
                            growth_allowance_db=growth,
                            zero_exclusion=zero_exclusion,
                        )

                        # Requested labels: 1 = oil; 0 = everything else.
                        binary_mask = (labels == 1).astype(np.uint8)

                        # Preserve which pixels were actually evaluated.
                        validity_mask = (labels != 255).astype(np.uint8)

                        created_paths.append(label_path)
                        save_binary_geotiff(label_path, binary_mask, src)

                        created_paths.append(validity_path)
                        save_binary_geotiff(validity_path, validity_mask, src)

                # Raster-aware copy preserves georeferencing and source masks.
                from rasterio.shutil import copy as copy_raster

                created_paths.append(image_path)
                copy_raster(source_path, image_path, driver="GTiff")

                report["source_filename"] = os.path.basename(source_path)
                report["mode"] = mode
                report["base_threshold_db"] = float(base_threshold_db)

                created_paths.append(report_path)
                with open(report_path, "w", encoding="utf-8") as report_file:
                    json.dump(report, report_file, indent=2)

                summary = report["object_summary"]

                writer.writerow({
                    "image_id": image_id,
                    "sar_path": image_path,
                    "label_path": label_path,
                    "estimated_oil_area_km2": (
                        report["estimated_oil_area_km2"]
                    ),
                    "num_objects": summary["num_objects"],
                    "num_high_confidence": summary["num_high_confidence"],
                    "num_medium_confidence": summary["num_medium_confidence"],
                    "num_low_confidence": summary["num_low_confidence"],
                    "validity_path": validity_path,
                    "report_path": report_path,
                })
                handle.flush()

                success_count += 1
                print(f"Saved image and label for: {image_id}")

            except Exception as exc:
                for path in created_paths:
                    if os.path.isfile(path):
                        os.remove(path)

                failure = {
                    "image_id": image_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                print(f"FAILED: {failure['error']}")

    with open(
        os.path.join(output_dir, "batch_failures.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(failures, handle, indent=2)

    print()
    print("Batch processing complete.")
    print("Successful images:", success_count)
    print("Failed images:", len(failures))
    print("Dataset index:", index_path)

    return {
        "index_path": index_path,
        "num_successful": success_count,
        "failures": failures,
    }




def generate_dataset_from_uploads(
    sar_files,
    worldcover_files,
    mode,
    base_threshold_db,
    zero_exclusion,
    processing_strategy="Auto",
    tile_size=2048,
    progress=None,
):
    """
    Copy uploaded files into local working folders, run the physics
    label generator, and return a downloadable dataset ZIP.
    """
    if not sar_files:
        return "Please upload at least one SAR GeoTIFF.", None

    if not worldcover_files:
        worldcover_files = [str(p) for p in (BASE_DIR / "worldcover_cache").glob("*.tif")]
    if not worldcover_files:
        return "No cached WorldCover references found. Add them to worldcover_cache or upload them.", None

    # The reference loader defined in cell 2 reads this setting.
    global WORLD_COVER_FOLDER, LATEST_DATASET_INDEX

    previous_reference_folder = WORLD_COVER_FOLDER
    log = io.StringIO()

    try:
        # A separate working directory prevents overwriting earlier datasets.
        work_folder = tempfile.mkdtemp(prefix="proteus_dataset_", dir=str(OUTPUT_DIR))

        sar_folder = os.path.join(work_folder, "sar_inputs")
        reference_folder = os.path.join(work_folder, "worldcover")
        dataset_folder = os.path.join(work_folder, "dataset")

        os.makedirs(sar_folder, exist_ok=True)
        os.makedirs(reference_folder, exist_ok=True)

        # Validate names before copying any files.
        sar_names = [
            os.path.basename(os.fspath(path))
            for path in sar_files
        ]
        reference_names = [
            os.path.basename(os.fspath(path))
            for path in worldcover_files
        ]

        sar_stems = [
            os.path.splitext(name)[0].casefold()
            for name in sar_names
        ]

        if len(set(sar_stems)) != len(sar_stems):
            raise ValueError(
                "Your SAR files contain duplicate filename stems. "
                "Give each SAR image a unique name."
            )

        if len(set(name.casefold() for name in reference_names)) != len(
            reference_names
        ):
            raise ValueError(
                "Your WorldCover uploads contain duplicate filenames."
            )

        for name in sar_names + reference_names:
            if not name.lower().endswith((".tif", ".tiff")):
                raise ValueError("Only .tif and .tiff files are accepted.")

        for path, name in zip(sar_files, sar_names):
            shutil.copy2(os.fspath(path), os.path.join(sar_folder, name))

        for path, name in zip(worldcover_files, reference_names):
            shutil.copy2(
                os.fspath(path),
                os.path.join(reference_folder, name),
            )

        WORLD_COVER_FOLDER = reference_folder

        with contextlib.redirect_stdout(log):
            result = batch_generate_oil_labels(
                input_dir=sar_folder,
                output_dir=dataset_folder,
                device_choice="CPU",
                mode=mode,
                base_threshold_db=float(base_threshold_db),
                zero_exclusion=bool(zero_exclusion),
                quick_preview=False,
                processing_strategy=processing_strategy,
                tile_size=int(tile_size),
                progress=progress,
            )

        if result["num_successful"] == 0:
            return (
                log.getvalue()
                + "\nNo dataset was generated. Check the errors above.",
                None,
            )

        # Make CSV paths relative so the downloaded dataset is portable.
        index_path = result["index_path"]
        LATEST_DATASET_INDEX = index_path

        with open(index_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)

        for row in rows:
            for column in [
                "sar_path",
                "label_path",
                "validity_path",
                "report_path",
            ]:
                row[column] = os.path.relpath(
                    row[column],
                    dataset_folder,
                ).replace(os.sep, "/")

        with open(
            index_path, "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        zip_path = shutil.make_archive(
            base_name=os.path.join(work_folder, "Proteus_training_dataset"),
            format="zip",
            root_dir=dataset_folder,
        )

        message = (
            log.getvalue()
            + "\nDataset ZIP is ready."
            + "\nCSV paths are relative to the extracted dataset folder."
            + "\nLabels are physics-generated suspected-oil pseudo-labels."
        )

        return message, zip_path

    except Exception as exc:
        return (
            log.getvalue()
            + f"\nERROR: {type(exc).__name__}: {exc}",
            None,
        )

    finally:
        WORLD_COVER_FOLDER = previous_reference_folder

def train_proteus_model(dataset_index, epochs=5, device_choice="CPU"):
    """Optional training; saves models/proteus_model.pt."""
    INDEX_PATH = os.path.abspath(dataset_index)
    DATASET_FOLDER = os.path.dirname(INDEX_PATH)
    EPOCHS = int(epochs)
    PATCH_SIZE = 256
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    if EPOCHS < 1:
        raise ValueError("Epochs must be at least 1.")
    torch.manual_seed(42)
    np.random.seed(42)
    def resolve_dataset_path(path):
        """Support both absolute paths and portable relative CSV paths."""
        if os.path.isabs(path):
            return path
        return os.path.join(DATASET_FOLDER, path)

    class OilTrainingDataset(Dataset):
        """Load SAR patches, suspected-oil labels, and evaluated-pixel masks."""

        def __init__(self, index_path, patch_size):
            self.patch_size = patch_size
            self.patches = []
            self.oil_pixels = 0
            self.non_oil_pixels = 0

            with open(index_path, newline="", encoding="utf-8") as handle:
                self.rows = list(csv.DictReader(handle))

            for row_index, row in enumerate(self.rows):
                sar_path = resolve_dataset_path(row["sar_path"])
                label_path = resolve_dataset_path(row["label_path"])
                valid_path = resolve_dataset_path(row["validity_path"])

                with rasterio.open(sar_path) as sar, \
                     rasterio.open(label_path) as labels, \
                     rasterio.open(valid_path) as validity:

                    for other in (labels, validity):
                        if (
                            other.width != sar.width
                            or other.height != sar.height
                            or other.crs != sar.crs
                            or not other.transform.almost_equals(sar.transform)
                        ):
                            raise ValueError(
                                f"Image and masks are not aligned: {row['image_id']}"
                            )

                    for top in range(0, sar.height, patch_size):
                        for left in range(0, sar.width, patch_size):
                            window = Window(
                                left,
                                top,
                                min(patch_size, sar.width - left),
                                min(patch_size, sar.height - top),
                            )

                            values = sar.read(1, window=window)
                            target = labels.read(1, window=window)
                            valid = (
                                (validity.read(1, window=window) == 1)
                                & np.isfinite(values)
                            )

                            if not valid.any():
                                continue

                            self.oil_pixels += int(
                                np.count_nonzero(valid & (target == 1))
                            )
                            self.non_oil_pixels += int(
                                np.count_nonzero(valid & (target == 0))
                            )

                            self.patches.append((row_index, window))

            if not self.patches:
                raise ValueError("No evaluated training patches were found.")

            if self.oil_pixels == 0 or self.non_oil_pixels == 0:
                raise ValueError(
                    "Training needs both suspected-oil pixels and evaluated "
                    "non-oil pixels. This dataset does not contain both."
                )

        def __len__(self):
            return len(self.patches)

        def __getitem__(self, index):
            row_index, window = self.patches[index]
            row = self.rows[row_index]

            with rasterio.open(resolve_dataset_path(row["sar_path"])) as src:
                image = src.read(1, window=window, out_dtype="float32")

            with rasterio.open(resolve_dataset_path(row["label_path"])) as src:
                target = src.read(1, window=window).astype(np.float32)

            with rasterio.open(resolve_dataset_path(row["validity_path"])) as src:
                valid = (
                    (src.read(1, window=window) == 1)
                    & np.isfinite(image)
                ).astype(np.float32)

            # Fixed normalization: use the same transformation for inference.
            image = np.where(np.isfinite(image), image, -40.0)
            image = (np.clip(image, -40.0, 5.0) + 40.0) / 45.0

            height, width = image.shape
            size = self.patch_size

            x = np.zeros((1, size, size), dtype=np.float32)
            y = np.zeros_like(x)
            weight = np.zeros_like(x)

            x[0, :height, :width] = image
            y[0, :height, :width] = target
            weight[0, :height, :width] = valid

            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(weight),
            )

    training_dataset = OilTrainingDataset(INDEX_PATH, PATCH_SIZE)

    device = torch.device("cuda" if device_choice == "Auto" and torch.cuda.is_available() else "cpu")

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    trained_model = ProteusSegNet().to(device)

    positive_weight = min(
        max(
            training_dataset.non_oil_pixels / training_dataset.oil_pixels,
            1.0,
        ),
        50.0,
    )

    loss_function = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([positive_weight], device=device),
    )

    optimizer = torch.optim.Adam(
        trained_model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Device:", device)

    print("Training patches:", len(training_dataset))

    print("Suspected-oil pixels:", training_dataset.oil_pixels)

    print("Evaluated non-oil pixels:", training_dataset.non_oil_pixels)

    print("Starting training...")

    training_history = []

    for epoch in range(1, EPOCHS + 1):
        trained_model.train()

        total_loss = 0.0
        total_valid = 0.0

        for images, targets, validity in training_loader:
            images = images.to(device)
            targets = targets.to(device)
            validity = validity.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = trained_model(images)
            pixel_loss = loss_function(logits, targets)

            valid_count = validity.sum()
            loss = (
                (pixel_loss * validity).sum()
                / valid_count.clamp_min(1)
            )

            loss.backward()
            optimizer.step()

            count = valid_count.item()
            total_loss += loss.item() * count
            total_valid += count

        average_loss = total_loss / max(total_valid, 1.0)
        training_history.append(average_loss)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Training loss: {average_loss:.6f}"
        )

    MODEL_FOLDER = str(MODEL_DIR)

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    MODEL_PATH = os.path.join(
        MODEL_FOLDER,
        "proteus_model.pt",
    )

    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in trained_model.state_dict().items()
            },
            "architecture": "ProteusSegNet",
            "initialization": "random",
            "label_source": "SAR physics pseudo-labels",
            "normalization": {
                "clip_min_db": -40.0,
                "clip_max_db": 5.0,
            },
            "patch_size": PATCH_SIZE,
            "epochs": EPOCHS,
            "positive_weight": positive_weight,
            "training_loss": training_history,
        },
        MODEL_PATH,
    )

    print()

    print("TRAINING COMPLETE.")

    print("Model saved:", MODEL_PATH)

    print("This model has not yet been independently validated.")
    return MODEL_PATH, training_history


# ==================================================
# 4. TILED PROCESSING FOR LARGE SAR SCENES
# ==================================================
#
# Scenes up to IN_MEMORY_PIXEL_LIMIT can still use the original full-scene
# path. Larger scenes are processed tile by tile so that peak RAM depends
# on the tile size, not on the scene size.
#
# The tiled result is equivalent to the full-scene algorithm:
#   * Local contrast uses a 51-pixel window, so every tile is read with a
#     halo of TILE_HALO (>= 25) pixels. Core pixels see exactly the same
#     neighbourhood as in a full-scene filter.
#   * Region growth (binary propagation of seeds inside the "allowed" mask
#     with 8-connectivity) is the same as keeping every 8-connected
#     component of "allowed" that contains at least one seed. Components
#     are labelled per tile and merged across tile seams with a sparse
#     connected-components pass, so slicks crossing tile borders remain one
#     object.
#   * As in the original code, objects (small-object removal, object IDs,
#     statistics) use 4-connectivity, the scipy.ndimage.label default, while
#     region growth uses 8-connectivity. Both are merged across seams with
#     the matching connectivity, so results are identical to the
#     full-scene path.
# Per-pixel labels are kept in a disk-backed scratch file (about 5 bytes
# per pixel), which is deleted after processing.

import gc
import time
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

IN_MEMORY_PIXEL_LIMIT = 16_000_000
MAX_TILED_PIXELS = 1_500_000_000
DEFAULT_TILE_SIZE = 2048
CONTRAST_WINDOW = 51
TILE_HALO = 32
MIN_OBJECT_PIXELS = 20
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
FOUR_CONNECTED = ndi.generate_binary_structure(2, 1)
PROCESSING_STRATEGIES = ["Auto", "Always tiled", "Full scene (≤16M px)"]


def should_use_tiling(width, height, strategy="Auto"):
    """Decide whether a scene is processed in tiles."""
    pixels = int(width) * int(height)
    if pixels > MAX_TILED_PIXELS:
        raise ValueError(
            f"The SAR image has {pixels:,} pixels. The tiled engine accepts "
            f"up to {MAX_TILED_PIXELS:,} pixels; crop the scene first."
        )
    if strategy == "Always tiled":
        return True
    if strategy == "Full scene (≤16M px)":
        if pixels > IN_MEMORY_PIXEL_LIMIT:
            raise ValueError(
                "Full-scene processing is limited to 16,000,000 pixels. "
                "Choose 'Auto' or 'Always tiled' for this image."
            )
        return False
    return pixels > IN_MEMORY_PIXEL_LIMIT


def aligned_tile_size(tile_size, patch_size=256):
    """Tile size rounded to a multiple of the model patch size."""
    tile_size = max(int(tile_size), int(patch_size))
    return (tile_size // int(patch_size)) * int(patch_size)


def iter_tiles(height, width, tile_size):
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            yield Window(
                left, top,
                min(tile_size, width - left),
                min(tile_size, height - top),
            )


def estimate_tiled_resources(width, height, tile_size=DEFAULT_TILE_SIZE):
    """Rough peak-RAM and scratch-disk estimates in megabytes."""
    tile_size = aligned_tile_size(tile_size)
    padded = (min(tile_size, height) + 2 * TILE_HALO) * (
        min(tile_size, width) + 2 * TILE_HALO
    )
    tiles = int(np.ceil(height / tile_size) * np.ceil(width / tile_size))
    return {
        "tiles": tiles,
        "tile_size": tile_size,
        "peak_ram_mb": round(padded * 80 / 1e6 + 450),
        "scratch_disk_mb": round(width * height * 5 / 1e6),
        "output_disk_mb": round(width * height * 2.5 / 1e6),
    }


class LocalWaterReference:
    """
    Windowed version of load_local_water_reference.

    Opens each local WorldCover tile once and reads only the window that
    is being processed. Classes: 0 = unknown, 1 = non-water, 2 = water.
    """

    def __init__(self, src):
        if src.crs is None:
            raise ValueError("The SAR image must have a CRS.")

        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds, densify_pts=21
        )
        if (
            not (-180 <= west < east <= 180)
            or not (-90 <= south < north <= 90)
            or east - west > 12
            or north - south > 12
        ):
            raise ValueError(
                "Use a regional SAR image no more than 12 degrees across "
                "and without antimeridian crossing."
            )

        self.sources, self.missing = [], []
        self._datasets, self._vrts = [], []
        self.known_classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

        latitudes = range(
            int(np.floor(south / 3) * 3),
            int(np.floor((north - 1e-9) / 3) * 3) + 1, 3,
        )
        longitudes = range(
            int(np.floor(west / 3) * 3),
            int(np.floor((east - 1e-9) / 3) * 3) + 1, 3,
        )

        for lat in latitudes:
            for lon in longitudes:
                filename = (
                    "ESA_WorldCover_10m_2021_v200_"
                    f"{worldcover_tile_name(lat, lon)}_Map.tif"
                )
                path = os.path.join(WORLD_COVER_FOLDER, filename)
                if not os.path.isfile(path):
                    self.missing.append(filename)
                    continue
                dataset = rasterio.open(path)
                self._datasets.append(dataset)
                self._vrts.append(WarpedVRT(
                    dataset,
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    src_nodata=0,
                    nodata=0,
                    resampling=Resampling.nearest,
                ))
                self.sources.append(path)

        if not self.sources:
            self.close()
            raise FileNotFoundError(
                "Required local WorldCover files were not found.\n"
                f"Folder: {WORLD_COVER_FOLDER}\n"
                "Expected files:\n" + "\n".join(self.missing)
            )

    def read(self, window):
        reference = np.zeros(
            (int(window.height), int(window.width)), dtype=np.uint8
        )
        for vrt in self._vrts:
            values = vrt.read(1, window=window)
            reference[np.isin(values, self.known_classes) & (values != 80)] = 1
            reference[values == 80] = 2
        return reference

    def close(self):
        for handle in self._vrts + self._datasets:
            try:
                handle.close()
            except Exception:
                pass
        self._vrts, self._datasets = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _seam_edges(label_map, height, width, tile_size, diagonal=True):
    """
    Pairs of labels that touch across tile seams (8-connectivity when
    diagonal is True, otherwise 4-connectivity).
    Only the two pixel lines on either side of each seam are read.
    """
    edges = []

    def collect(a, b):
        for shift in ((-1, 0, 1) if diagonal else (0,)):
            if shift < 0:
                x, y = a[-shift:], b[:shift]
            elif shift > 0:
                x, y = a[:-shift], b[shift:]
            else:
                x, y = a, b
            both = (x > 0) & (y > 0)
            if both.any():
                edges.append(np.stack([x[both], y[both]], axis=1))

    for row in range(tile_size, height, tile_size):
        collect(np.asarray(label_map[row - 1, :]), np.asarray(label_map[row, :]))
    for col in range(tile_size, width, tile_size):
        collect(np.asarray(label_map[:, col - 1]), np.asarray(label_map[:, col]))

    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(edges).astype(np.int64), axis=0)


def _merge_components(label_map, num_labels, height, width, tile_size,
                      diagonal=True):
    """Map every tile-local label to a scene-wide component index."""
    edges = _seam_edges(label_map, height, width, tile_size, diagonal)
    graph = coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
        shape=(num_labels + 1, num_labels + 1),
    )
    _, component = connected_components(graph, directed=False)
    return component


def _make_scratch_dir():
    """
    Temporary folder for the disk-backed label maps. Uses the system temp
    folder (not the outputs folder, which may be synced by OneDrive and slow
    down in-place writes). Override with the PROTEUS_SCRATCH_DIR variable.
    """
    root = os.environ.get("PROTEUS_SCRATCH_DIR") or tempfile.gettempdir()
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix="proteus_tiling_", dir=root)


def _remove_scratch(path, attempts=5):
    """Delete a scratch folder; retry because Windows/OneDrive may hold it briefly."""
    for attempt in range(attempts):
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.5 * (attempt + 1))


def _report_progress(progress, fraction, message):
    if progress is not None:
        try:
            progress(float(min(max(fraction, 0.0), 1.0)), desc=message)
        except Exception:
            pass


def _model_scores_for_tile(model, data, evaluated, patch_size,
                           minimum_db, maximum_db, device):
    """Patch-wise ProteusSegNet inference, identical to the full-scene loop."""
    height, width = data.shape
    scores = np.full(data.shape, -9999.0, dtype=np.float32)

    with torch.inference_mode():
        for top in range(0, height, patch_size):
            for left in range(0, width, patch_size):
                h = min(patch_size, height - top)
                w = min(patch_size, width - left)
                tile_valid = evaluated[top:top + h, left:left + w]
                if not tile_valid.any():
                    continue

                tile = data[top:top + h, left:left + w]
                tile = np.where(np.isfinite(tile), tile, minimum_db)
                normalized = (
                    np.clip(tile, minimum_db, maximum_db) - minimum_db
                ) / (maximum_db - minimum_db)

                padded = np.zeros((patch_size, patch_size), dtype=np.float32)
                padded[:h, :w] = normalized
                tensor = torch.from_numpy(padded)[None, None].to(device)
                prediction = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()[:h, :w]

                target = scores[top:top + h, left:left + w]
                target[tile_valid] = prediction[tile_valid]
    return scores


def tiled_detection(
    src,
    threshold_db,
    contrast_min,
    growth_allowance_db,
    zero_exclusion,
    work_dir,
    model=None,
    model_config=None,
    model_threshold=0.5,
    scores_path=None,
    tile_size=DEFAULT_TILE_SIZE,
    min_pixels=MIN_OBJECT_PIXELS,
    progress=None,
    progress_span=(0.0, 0.7),
):
    """
    Passes 1-2 of the tiled engine.

    Pass 1: per tile, SAR physics (water mask, contrast, seeds, growth mask),
            optional model scores, tile-local 8-connected labelling of the
            growth mask.
    Merge:  join labels across seams -> grown regions (seeded components).
    Pass 2: per tile, intersect grown regions with the model threshold and
            relabel with 4-connectivity; merge again; drop objects below
            min_pixels. (Removing small physics objects first, as the
            full-scene code does, cannot change this result: every final
            object lies inside one physics object.)

    Returns a dict describing the final objects. The per-pixel object map
    is available through ``final_ids(window)``.
    """
    if src.count != 1:
        raise ValueError("The SAR input must contain exactly one band.")

    height, width = src.height, src.width
    patch_size = int(model_config["patch_size"]) if model_config else 256
    tile_size = aligned_tile_size(tile_size, patch_size)
    windows = list(iter_tiles(height, width, tile_size))
    start, end = progress_span
    span = end - start

    needed = height * width * 5
    free = shutil.disk_usage(work_dir).free
    if needed > free * 0.9:
        raise OSError(
            f"Tiled processing needs about {needed / 1e9:.1f} GB of scratch "
            f"disk space in {work_dir}; only {free / 1e9:.1f} GB is free."
        )

    label_map = np.lib.format.open_memmap(
        os.path.join(work_dir, "_labels.npy"), mode="w+",
        dtype=np.int32, shape=(height, width),
    )
    flags = np.lib.format.open_memmap(
        os.path.join(work_dir, "_flags.npy"), mode="w+",
        dtype=np.uint8, shape=(height, width),
    )
    WATER, MODEL_PASS = 1, 2

    sizes, has_seed = [np.zeros(1, np.int64)], [np.zeros(1, bool)]
    next_label = 1
    water_pixels = 0
    data_min, data_max = np.inf, -np.inf

    scores_dst = None
    if model is not None and scores_path:
        scores_dst = rasterio.open(
            scores_path, "w", driver="GTiff", height=height, width=width,
            count=1, dtype="float32", nodata=-9999.0, crs=src.crs,
            transform=src.transform, compress="deflate", tiled=True,
            blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER",
        )

    try:
        with LocalWaterReference(src) as reference:
            # ---------------- Pass 1 ----------------
            for number, window in enumerate(windows, start=1):
                row0, col0 = int(window.row_off), int(window.col_off)
                h, w = int(window.height), int(window.width)
                r0, c0 = max(0, row0 - TILE_HALO), max(0, col0 - TILE_HALO)
                r1 = min(height, row0 + h + TILE_HALO)
                c1 = min(width, col0 + w + TILE_HALO)
                padded = Window(c0, r0, c1 - c0, r1 - r0)

                data_p = src.read(1, window=padded, out_dtype="float32")
                valid_p = (src.read_masks(1, window=padded) > 0) & np.isfinite(data_p)
                if zero_exclusion:
                    valid_p &= data_p != 0
                water_p = valid_p & (reference.read(padded) == 2)

                core = (slice(row0 - r0, row0 - r0 + h), slice(col0 - c0, col0 - c0 + w))
                data = data_p[core]
                water = water_p[core]

                if water.any():
                    contrast = calculate_local_contrast(
                        data_p, water_p, window_size=CONTRAST_WINDOW
                    )[core]
                    seeds = water & (data < threshold_db) & (contrast > contrast_min)
                    allowed = water & (data < threshold_db + growth_allowance_db)
                    del contrast
                    water_values = data[water]
                    data_min = min(data_min, float(water_values.min()))
                    data_max = max(data_max, float(water_values.max()))
                    water_pixels += int(water.sum())
                else:
                    seeds = allowed = np.zeros_like(water)

                local, count = ndi.label(allowed, structure=EIGHT_CONNECTED)
                if count:
                    if next_label + count >= np.iinfo(np.int32).max:
                        raise ValueError("Too many candidate regions; use a smaller scene.")
                    sizes.append(np.bincount(local.ravel(), minlength=count + 1)[1:])
                    has_seed.append(np.bincount(local[seeds], minlength=count + 1)[1:] > 0)
                    local[local > 0] += next_label - 1
                    next_label += count
                label_map[row0:row0 + h, col0:col0 + w] = local

                tile_flags = water.astype(np.uint8) * WATER
                if model is None:
                    tile_flags |= MODEL_PASS
                else:
                    scores = _model_scores_for_tile(
                        model, data, water, patch_size,
                        model_config["minimum_db"], model_config["maximum_db"],
                        model_config["device"],
                    )
                    tile_flags[scores >= float(model_threshold)] |= MODEL_PASS
                    if scores_dst is not None:
                        scores_dst.write(scores, 1, window=window)
                flags[row0:row0 + h, col0:col0 + w] = tile_flags

                _report_progress(
                    progress, start + span * 0.6 * number / len(windows),
                    f"Tile {number}/{len(windows)}: SAR physics"
                    + (" + ProteusSegNet" if model is not None else ""),
                )

            reference_sources = list(reference.sources)
            missing_references = list(reference.missing)
    finally:
        if scores_dst is not None:
            scores_dst.close()

    if water_pixels == 0:
        raise ValueError("No valid reference-water pixels were found.")

    # ---------------- Merge physics components ----------------
    num_labels = next_label - 1
    has_seed = np.concatenate(has_seed)
    component = _merge_components(label_map, num_labels, height, width, tile_size)
    component_seed = np.bincount(component, weights=has_seed) > 0
    physics_keep = component_seed[component]
    physics_keep[0] = False
    del sizes, has_seed, component, component_seed
    _report_progress(progress, start + span * 0.7, "Merging regions across tile seams")

    # ---------------- Pass 2 ----------------
    sizes2 = [np.zeros(1, np.int64)]
    next_label = 1
    physics_pixels = 0
    for number, window in enumerate(windows, start=1):
        row0, col0 = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        physics = physics_keep[label_map[row0:row0 + h, col0:col0 + w]]
        physics_pixels += int(physics.sum())
        candidate = physics & ((flags[row0:row0 + h, col0:col0 + w] & MODEL_PASS) > 0)

        local, count = ndi.label(candidate, structure=FOUR_CONNECTED)
        if count:
            sizes2.append(np.bincount(local.ravel(), minlength=count + 1)[1:])
            local[local > 0] += next_label - 1
            next_label += count
        label_map[row0:row0 + h, col0:col0 + w] = local
        _report_progress(
            progress, start + span * (0.7 + 0.25 * number / len(windows)),
            f"Tile {number}/{len(windows)}: combining physics and model",
        )
    del physics_keep

    num_labels = next_label - 1
    sizes2 = np.concatenate(sizes2)
    component = _merge_components(label_map, num_labels, height, width,
                                  tile_size, diagonal=False)
    component_size = np.bincount(component, weights=sizes2)
    keep = component_size >= min_pixels
    keep[component[0]] = False

    # Number the surviving objects 1..N in tile-scan order.
    kept_components = np.unique(component[1:][keep[component[1:]]])
    final_of_component = np.zeros(component.max() + 1, dtype=np.int32)
    final_of_component[kept_components] = np.arange(1, len(kept_components) + 1)
    final_of_label = final_of_component[component]
    final_of_label[0] = 0
    _report_progress(progress, end, "Scene-wide objects resolved")

    def final_ids(window):
        row0, col0 = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        return final_of_label[label_map[row0:row0 + h, col0:col0 + w]]

    def water_mask(window):
        row0, col0 = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        return (flags[row0:row0 + h, col0:col0 + w] & WATER) > 0

    def release():
        nonlocal label_map, flags
        label_map = flags = None
        gc.collect()
        for name in ("_labels.npy", "_flags.npy"):
            try:
                os.remove(os.path.join(work_dir, name))
            except OSError:
                pass

    return {
        "windows": windows,
        "tile_size": tile_size,
        "num_objects": int(len(kept_components)),
        "water_pixels": water_pixels,
        "physics_oil_pixels": physics_pixels,
        "data_min": data_min,
        "data_max": data_max,
        "reference_sources": reference_sources,
        "missing_references": missing_references,
        "final_ids": final_ids,
        "water_mask": water_mask,
        "release": release,
    }


def tiled_object_statistics(src, detection, scores_path=None,
                            tile_writer=None, preview_size=1200,
                            progress=None, progress_span=(0.7, 0.85)):
    """
    Pass 3: per-object area, mean backscatter, mean model score, bounding
    boxes, background statistics, histogram counts and a preview image.
    ``tile_writer(window, ids, water)`` can write per-tile output rasters.
    """
    height, width = src.height, src.width
    count = detection["num_objects"]
    areas_by_row = pixel_area_by_row(src)

    pixel_count = np.zeros(count + 1, np.int64)
    area = np.zeros(count + 1, np.float64)
    sum_db = np.zeros(count + 1, np.float64)
    sum_score = np.zeros(count + 1, np.float64)
    tiles_touched = np.zeros(count + 1, np.int32)
    row_min = np.full(count + 1, height, np.int64)
    row_max = np.zeros(count + 1, np.int64)
    col_min = np.full(count + 1, width, np.int64)
    col_max = np.zeros(count + 1, np.int64)
    background_sum, background_count = 0.0, 0

    low, high = detection["data_min"], detection["data_max"]
    if not high > low:
        high = low + 1
    bins = np.linspace(low, high, 61)
    hist_background = np.zeros(60, np.int64)
    hist_oil = np.zeros(60, np.int64)

    step = max(1, int(np.ceil(max(height, width) / preview_size)))
    preview = np.full((-(-height // step), -(-width // step)), np.nan, np.float32)
    preview_labels = np.full(preview.shape, 255, np.uint8)

    scores_src = rasterio.open(scores_path) if scores_path else None
    windows = detection["windows"]
    try:
        for number, window in enumerate(windows, start=1):
            row0, col0 = int(window.row_off), int(window.col_off)
            h, w = int(window.height), int(window.width)
            ids = detection["final_ids"](window)
            water = detection["water_mask"](window)
            oil = ids > 0
            data = src.read(1, window=window, out_dtype="float32")

            if tile_writer is not None:
                tile_writer(window, ids, water)

            if oil.any():
                flat = ids.ravel()
                pixel_count += np.bincount(flat, minlength=count + 1)
                sum_db += np.bincount(flat, weights=data.ravel().astype(np.float64), minlength=count + 1)
                if areas_by_row is not None:
                    row_weights = np.repeat(areas_by_row[row0:row0 + h] / 1_000_000, w)
                    area += np.bincount(flat, weights=row_weights, minlength=count + 1)
                if scores_src is not None:
                    scores = scores_src.read(1, window=window)
                    sum_score += np.bincount(
                        flat, weights=np.where(oil, scores, 0).ravel().astype(np.float64),
                        minlength=count + 1,
                    )
                present = np.unique(ids[oil])
                tiles_touched[present] += 1
                slices = ndi.find_objects(ids, max_label=int(present.max()))
                for object_id in present:
                    rows, cols = slices[object_id - 1]
                    row_min[object_id] = min(row_min[object_id], row0 + rows.start)
                    row_max[object_id] = max(row_max[object_id], row0 + rows.stop)
                    col_min[object_id] = min(col_min[object_id], col0 + cols.start)
                    col_max[object_id] = max(col_max[object_id], col0 + cols.stop)

            background = water & ~oil
            if background.any():
                values = data[background].astype(np.float64)
                background_sum += float(values.sum())
                background_count += int(values.size)
                hist_background += np.histogram(values, bins=bins)[0]
            if oil.any():
                hist_oil += np.histogram(data[oil], bins=bins)[0]

            # Preview on a global grid so tiles join seamlessly.
            r_start, c_start = (-row0) % step, (-col0) % step
            sub = (slice(r_start, h, step), slice(c_start, w, step))
            pr, pc = (row0 + r_start) // step, (col0 + c_start) // step
            sub_data = data[sub]
            sub_labels = np.full(sub_data.shape, 255, np.uint8)
            sub_labels[water[sub]] = 0
            sub_labels[oil[sub]] = 1
            preview[pr:pr + sub_data.shape[0], pc:pc + sub_data.shape[1]] = sub_data
            preview_labels[pr:pr + sub_data.shape[0], pc:pc + sub_data.shape[1]] = sub_labels

            start, end = progress_span
            _report_progress(progress, start + (end - start) * number / len(windows),
                             f"Tile {number}/{len(windows)}: object statistics")
    finally:
        if scores_src is not None:
            scores_src.close()

    mean_background = background_sum / background_count if background_count else None
    rows = []
    for object_id in range(1, count + 1):
        object_area = float(area[object_id]) if areas_by_row is not None else None
        mean_db = float(sum_db[object_id] / pixel_count[object_id])
        contrast_db = mean_background - mean_db if mean_background is not None else None
        if contrast_db is None:
            confidence = "unknown"
        elif contrast_db >= 4 and (object_area is None or object_area >= 0.1):
            confidence = "high"
        elif contrast_db >= 2 and (object_area is None or object_area >= 0.02):
            confidence = "medium"
        else:
            confidence = "low"
        rows.append({
            "object_id": object_id,
            "pixel_count": int(pixel_count[object_id]),
            "area_km2": object_area,
            "mean_backscatter_db": mean_db,
            "contrast_vs_water_db": contrast_db,
            "mean_model_score": (
                float(sum_score[object_id] / pixel_count[object_id])
                if scores_path else None
            ),
            "heuristic_confidence": confidence,
        })

    estimated_area = float(area[1:].sum()) if areas_by_row is not None else None
    return {
        "object_rows": rows,
        "estimated_area_km2": estimated_area,
        "oil_pixels": int(pixel_count[1:].sum()),
        "mean_background_db": mean_background,
        "bboxes": (row_min, row_max, col_min, col_max),
        "tiles_touched": tiles_touched,
        "hist_bins": bins,
        "hist_background": hist_background,
        "hist_oil": hist_oil,
        "preview": preview,
        "preview_labels": preview_labels,
    }


def tiled_polygonize(src, ids_path, geojson_path, stats, windows,
                     progress=None, progress_span=(0.85, 0.95)):
    """
    Stream object polygons to GeoJSON without holding the scene in memory.
    Objects inside one tile are polygonized per tile; objects crossing tile
    seams are polygonized from a window around their bounding box.
    """
    properties = {row["object_id"]: row for row in stats["object_rows"]}
    tiles_touched = stats["tiles_touched"]
    row_min, row_max, col_min, col_max = stats["bboxes"]
    single_tile = tiles_touched == 1
    first = True

    def emit(handle, geometry, value):
        nonlocal first
        feature = {
            "type": "Feature",
            "geometry": transform_geom(src.crs, "EPSG:4326", geometry),
            "properties": properties[int(value)],
        }
        handle.write(("" if first else ",\n") + json.dumps(feature))
        first = False

    with rasterio.open(ids_path) as ids_src, \
            open(geojson_path, "w", encoding="utf-8") as handle:
        handle.write('{"type": "FeatureCollection", "features": [\n')
        start, end = progress_span

        for number, window in enumerate(windows, start=1):
            ids = ids_src.read(1, window=window)
            mask = (ids > 0) & single_tile[ids]
            if mask.any():
                for geometry, value in shapes(
                    ids, mask=mask, transform=window_transform(window, src.transform)
                ):
                    emit(handle, geometry, value)
            _report_progress(progress, start + (end - start) * 0.8 * number / len(windows),
                             f"Tile {number}/{len(windows)}: polygons")

        for object_id in np.flatnonzero(tiles_touched > 1):
            window = Window(
                int(col_min[object_id]), int(row_min[object_id]),
                int(col_max[object_id] - col_min[object_id]),
                int(row_max[object_id] - row_min[object_id]),
            )
            ids = ids_src.read(1, window=window)
            for geometry, value in shapes(
                ids, mask=ids == object_id,
                transform=window_transform(window, src.transform),
            ):
                emit(handle, geometry, value)
        handle.write("\n]}\n")
    _report_progress(progress, progress_span[1], "Polygons exported")


def tiled_physics_labels_to_files(src, threshold_db, contrast_min,
                                  growth_allowance_db, zero_exclusion,
                                  label_path, validity_path,
                                  tile_size=DEFAULT_TILE_SIZE, progress=None):
    """
    Tiled counterpart of create_physics_labels for dataset generation.
    Writes the binary oil mask and validity mask directly to GeoTIFF and
    returns the same report structure as create_physics_labels.
    """
    work_dir = _make_scratch_dir()
    try:
        detection = tiled_detection(
            src, threshold_db, contrast_min, growth_allowance_db,
            zero_exclusion, work_dir, tile_size=tile_size, progress=progress,
        )
    except Exception:
        _remove_scratch(work_dir)
        raise
    profile = {
        "driver": "GTiff", "height": src.height, "width": src.width,
        "count": 1, "dtype": "uint8", "crs": src.crs,
        "transform": src.transform, "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
        "nodata": None,
    }
    try:
        with rasterio.open(label_path, "w", **profile) as label_dst, \
                rasterio.open(validity_path, "w", **profile) as valid_dst:
            def writer(window, ids, water):
                label_dst.write((ids > 0).astype(np.uint8), 1, window=window)
                valid_dst.write(water.astype(np.uint8), 1, window=window)

            stats = tiled_object_statistics(
                src, detection, tile_writer=writer, progress=progress,
            )
    finally:
        detection["release"]()
        _remove_scratch(work_dir)

    counts = {level: 0 for level in ["high", "medium", "low", "unknown"]}
    for row in stats["object_rows"]:
        counts[row["heuristic_confidence"]] += 1

    total = src.width * src.height
    return {
        "model": "Local SAR physics pseudo-labeler",
        "processing": f"tiled ({detection['tile_size']} px tiles, "
                      f"{len(detection['windows'])} tiles)",
        "estimated_oil_area_km2": stats["estimated_area_km2"],
        "selected_oil_pixels": stats["oil_pixels"],
        "evaluated_pixels": detection["water_pixels"],
        "excluded_pixels": int(total - detection["water_pixels"]),
        "threshold_db": float(threshold_db),
        "contrast_min": float(contrast_min),
        "growth_allowance_db": float(growth_allowance_db),
        "zero_exclusion": bool(zero_exclusion),
        "reference_sources": detection["reference_sources"],
        "object_summary": {
            "num_objects": len(stats["object_rows"]),
            "num_high_confidence": counts["high"],
            "num_medium_confidence": counts["medium"],
            "num_low_confidence": counts["low"],
            "num_unknown_confidence": counts["unknown"],
        },
    }


# ==================================================
# PROJECT PROTEUS — MAIN PROCESSING PIPELINE
# Trained model + local reference + SAR physics
# ==================================================

import os
import csv
import json
import tempfile
import zipfile

import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import transform_geom


TITLE = "Project Proteus: Mapping of Suspected Oil Slicks using SAR & GeoAI"

LATEST_PROTEUS_ZIP = None



def analysis_limitations():
    return [
        "Trained using physics-generated pseudo-labels.",
        "Independent validation has not been completed.",
        "Model scores are not calibrated oil probabilities.",
        "Confidence classes are heuristic, not verified certainty.",
    ]


def save_analysis_figures(results_folder, preview, preview_labels,
                          display_range, bins, hist_background, hist_oil,
                          object_rows):
    """Create the three analytical figures used by the app and the ZIP."""
    from matplotlib.patches import Patch

    graph_paths = []
    low, high = (float(v) for v in display_range)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1
    preview_valid = (preview_labels != 255) & np.isfinite(preview)

    # Figure 1: SAR context and suspected-oil delineation.
    overlay = np.zeros((*preview.shape, 4), dtype=np.float32)
    overlay[preview_labels == 1] = [0.86, 0.1, 0.1, 0.7]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    try:
        for axis in axes:
            axis.set_facecolor("#e9edf1")
            axis.imshow(np.ma.array(preview, mask=~preview_valid),
                        cmap="gray", vmin=low, vmax=high)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[0].set_title("SAR backscatter within evaluated water")
        axes[1].imshow(overlay)
        axes[1].set_title("Integrated suspected-oil delineation")
        figure.legend(
            handles=[
                Patch(color="#db1a1a", alpha=0.7, label="Suspected oil"),
                Patch(color="#9a9a9a", label="Evaluated water (SAR)"),
                Patch(facecolor="#e9edf1", edgecolor="#9aa5b1",
                      label="Excluded (land / unknown / no data)"),
            ],
            loc="lower center", ncol=3, frameon=False,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 1))
        path = os.path.join(results_folder, "oil_preview.png")
        figure.savefig(path, dpi=150)
        graph_paths.append(path)
    finally:
        plt.close(figure)

    # Figure 2: backscatter distributions (probability density).
    figure, axis = plt.subplots(figsize=(9, 5))
    try:
        widths = np.diff(bins)
        for counts, name, color in [
            (hist_background, "Evaluated non-oil water", "tab:blue"),
            (hist_oil, "Suspected oil", "tab:red"),
        ]:
            total = counts.sum()
            if total:
                axis.bar(bins[:-1], counts / (total * widths), width=widths,
                         align="edge", alpha=0.5, label=name, color=color)
        axis.set_xlabel("SAR backscatter (dB)")
        axis.set_ylabel("Probability density")
        axis.set_title("Backscatter distribution by predicted class")
        axis.legend()
        figure.tight_layout()
        path = os.path.join(results_folder, "backscatter_histogram.png")
        figure.savefig(path, dpi=150)
        graph_paths.append(path)
    finally:
        plt.close(figure)

    # Figure 3: object area versus physical contrast.
    figure, axis = plt.subplots(figsize=(9, 5))
    try:
        plotted = False
        for level, color in [("high", "green"), ("medium", "orange"),
                             ("low", "red"), ("unknown", "gray")]:
            rows = [
                row for row in object_rows
                if row["heuristic_confidence"] == level
                and row["area_km2"] is not None
                and row["contrast_vs_water_db"] is not None
                and row["area_km2"] > 0
            ]
            if rows:
                axis.scatter([row["area_km2"] for row in rows],
                             [row["contrast_vs_water_db"] for row in rows],
                             label=f"{level} ({len(rows)})", color=color, s=14)
                plotted = True
        if plotted:
            axis.set_xscale("log")
            axis.legend(title="Heuristic confidence")
        else:
            axis.text(0.5, 0.5, "No objects with available area and contrast.",
                      transform=axis.transAxes, ha="center")
        axis.set_xlabel("Object area (km²)")
        axis.set_ylabel("Background-water mean − object mean (dB)")
        axis.set_title("Detected object area and backscatter contrast")
        figure.tight_layout()
        path = os.path.join(results_folder, "object_statistics.png")
        figure.savefig(path, dpi=150)
        graph_paths.append(path)
    finally:
        plt.close(figure)

    return graph_paths


def format_analysis_summary(report):
    area = report.get("estimated_oil_area_km2")
    area_text = (f"{area:.4f} km²" if area is not None
                 else "Unavailable for this CRS geometry")
    processing = report.get("processing", {})
    summary = report["object_summary"]
    return (
        "Project Proteus | Analysis completed\n\n"
        f"Source: {report['source_filename']}\n"
        f"Processing: {processing.get('strategy', 'full scene')}\n"
        f"Runtime: {processing.get('runtime_seconds', 0):.1f} s\n"
        f"Device: {report['device']}\n"
        f"Mode: {report['mode']}\n"
        f"Effective SAR threshold: {report['threshold_db']:.2f} dB\n"
        f"Model score threshold: {report['model_threshold']:.2f}\n"
        f"Suspected-oil pixels: {report['selected_oil_pixels']:,}\n"
        f"Estimated suspected-oil area: {area_text}\n"
        f"Objects: {summary['num_objects']} "
        f"(high {summary['num_high_confidence']}, "
        f"medium {summary['num_medium_confidence']}, "
        f"low {summary['num_low_confidence']})\n"
        f"Excluded pixels: {report['excluded_pixels']:,}\n\n"
        "Final detections satisfy the ProteusSegNet score threshold and SAR physics criteria.\n"
        "Missing reference coverage remains excluded.\n"
        "Independent validation has not been completed."
    )


def finish_analysis(work_folder, results_folder, report, graph_paths, processing):
    """Write report.json, package the results ZIP and build the summary."""
    global LATEST_PROTEUS_ZIP
    report["processing"] = processing
    with open(os.path.join(results_folder, "report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    zip_path = shutil.make_archive(
        os.path.join(work_folder, "Proteus_results"), "zip", results_folder
    )
    LATEST_PROTEUS_ZIP = zip_path
    return format_analysis_summary(report), graph_paths, zip_path


def run_tiled_analysis(src, sar_name, work_folder, results_folder, model,
                       model_config, mode, threshold_db, contrast_min,
                       growth_allowance_db, model_threshold, zero_exclusion,
                       tile_size, started, progress=None):
    """Integrated physics + ProteusSegNet analysis for large scenes."""
    scratch = _make_scratch_dir()
    scores_path = os.path.join(results_folder, "model_scores.tif")
    labels_path = os.path.join(results_folder, "suspected_oil_labels.tif")
    ids_path = os.path.join(results_folder, "oil_object_ids.tif")
    detection = None
    try:
        _report_progress(progress, 0.02, "Preparing tiled processing")
        detection = tiled_detection(
            src, threshold_db, contrast_min, growth_allowance_db,
            zero_exclusion, scratch, model=model, model_config=model_config,
            model_threshold=model_threshold, scores_path=scores_path,
            tile_size=tile_size, progress=progress, progress_span=(0.02, 0.7),
        )
        profile = {
            "driver": "GTiff", "height": src.height, "width": src.width,
            "count": 1, "crs": src.crs, "transform": src.transform,
            "compress": "deflate", "tiled": True, "blockxsize": 256,
            "blockysize": 256, "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(labels_path, "w", dtype="uint8", nodata=255,
                           **profile) as labels_dst, \
                rasterio.open(ids_path, "w", dtype="int32", nodata=0,
                              **profile) as ids_dst:
            def writer(window, ids, water):
                labels = np.full(ids.shape, 255, dtype=np.uint8)
                labels[water] = 0
                labels[ids > 0] = 1
                labels_dst.write(labels, 1, window=window)
                ids_dst.write(ids.astype(np.int32), 1, window=window)

            stats = tiled_object_statistics(
                src, detection, scores_path=scores_path, tile_writer=writer,
                progress=progress, progress_span=(0.7, 0.85),
            )
    finally:
        if detection is not None:
            detection["release"]()
        _remove_scratch(scratch)

    tiled_polygonize(
        src, ids_path, os.path.join(results_folder, "oil_objects.geojson"),
        stats, detection["windows"], progress=progress,
    )

    object_rows = stats["object_rows"]
    total = src.width * src.height
    report = {
        "project": TITLE,
        "source_filename": sar_name,
        "model": "ProteusSegNet",
        "device": str(model_config["device"]),
        "mode": mode,
        "threshold_db": float(threshold_db),
        "contrast_min": float(contrast_min),
        "growth_allowance_db": float(growth_allowance_db),
        "model_threshold": float(model_threshold),
        "estimated_oil_area_km2": stats["estimated_area_km2"],
        "selected_oil_pixels": stats["oil_pixels"],
        "evaluated_water_pixels": detection["water_pixels"],
        "excluded_pixels": int(total - detection["water_pixels"]),
        "object_summary": {
            "num_objects": len(object_rows),
            **{
                f"num_{level}_confidence": sum(
                    row["heuristic_confidence"] == level for row in object_rows
                )
                for level in ["high", "medium", "low", "unknown"]
            },
        },
        "reference_files": [
            os.path.basename(path) for path in detection["reference_sources"]
        ],
        "limitations": analysis_limitations(),
    }

    columns = ["object_id", "pixel_count", "area_km2", "mean_backscatter_db",
               "contrast_vs_water_db", "mean_model_score",
               "heuristic_confidence"]
    with open(os.path.join(results_folder, "oil_objects.csv"), "w",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(object_rows)

    _report_progress(progress, 0.96, "Rendering figures")
    preview = stats["preview"]
    preview_labels = stats["preview_labels"]
    preview_values = preview[(preview_labels != 255) & np.isfinite(preview)]
    display_range = (
        np.percentile(preview_values, [2, 98]) if preview_values.size else (0, 1)
    )
    graph_paths = save_analysis_figures(
        results_folder, preview, preview_labels, display_range,
        stats["hist_bins"], stats["hist_background"], stats["hist_oil"],
        object_rows,
    )
    _report_progress(progress, 0.99, "Packaging results")
    return finish_analysis(
        work_folder, results_folder, report, graph_paths,
        processing={
            "strategy": (
                f"tiled ({detection['tile_size']} px tiles, "
                f"{len(detection['windows'])} tiles, {TILE_HALO} px halo)"
            ),
            "scene_pixels": int(total),
            "tile_size": detection["tile_size"],
            "tiles": len(detection["windows"]),
            "runtime_seconds": round(time.time() - started, 1),
        },
    )


def run_proteus_project(
    sar_file,
    reference_files,
    model_file,
    mode,
    base_threshold_db,
    model_threshold,
    zero_exclusion,
    device_choice,
    processing_strategy="Auto",
    tile_size=2048,
    progress=None,
):
    """
    Run the integrated Proteus pipeline.

    Scenes above 16 million pixels (or any scene when processing_strategy
    is "Always tiled") are processed with the memory-bounded tiled engine.

    Final detections must satisfy:
    1. Valid SAR and local reference-water coverage.
    2. SAR physics pseudo-label criteria.
    3. The trained model's score threshold.

    Model scores and object confidence classes are not calibrated
    probabilities of a real oil spill.
    """
    global WORLD_COVER_FOLDER, LATEST_PROTEUS_ZIP

    started = time.time()
    if not sar_file:
        return "Please upload a SAR GeoTIFF.", [], None

    reference_files = reference_files or []

    checkpoint_path = (
        os.fspath(model_file)
        if model_file
        else str(MODEL_DIR / "proteus_model.pt")
    )

    if not os.path.isfile(checkpoint_path):
        return (
            "Upload your trained proteus_model.pt checkpoint.",
            [],
            None,
        )

    previous_reference_folder = WORLD_COVER_FOLDER
    graph_paths = []

    try:
        work_folder = tempfile.mkdtemp(prefix="proteus_main_", dir=str(OUTPUT_DIR))
        reference_folder = str(BASE_DIR / "worldcover_cache")
        results_folder = os.path.join(work_folder, "results")

        os.makedirs(reference_folder, exist_ok=True)
        os.makedirs(results_folder)

        reference_names = [
            os.path.basename(os.fspath(path))
            for path in reference_files
        ]

        if len(set(name.casefold() for name in reference_names)) != len(
            reference_names
        ):
            raise ValueError("Duplicate WorldCover filenames were uploaded.")

        for path, name in zip(reference_files, reference_names):
            shutil.copy2(
                os.fspath(path),
                os.path.join(reference_folder, name),
            )

        WORLD_COVER_FOLDER = reference_folder

        threshold_db, contrast_min, growth_allowance_db = (
            get_operational_params(mode, base_threshold_db)
        )

        # Load only the checkpoint format created by our training cell.
        saved = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if saved.get("architecture") != "ProteusSegNet":
            raise ValueError("This app expects a ProteusSegNet checkpoint.")

        device = torch.device(
            "cuda"
            if device_choice == "Auto" and torch.cuda.is_available()
            else "cpu"
        )

        model = ProteusSegNet()
        model.load_state_dict(saved["model_state_dict"])
        model.to(device)
        model.eval()

        patch_size = int(saved["patch_size"])
        minimum_db = float(saved["normalization"]["clip_min_db"])
        maximum_db = float(saved["normalization"]["clip_max_db"])

        with rasterio.open(os.fspath(sar_file)) as src:
            if src.count != 1:
                raise ValueError("The SAR input must contain exactly one band.")
            if should_use_tiling(src.width, src.height, processing_strategy):
                return run_tiled_analysis(
                    src,
                    sar_name=os.path.basename(os.fspath(sar_file)),
                    work_folder=work_folder,
                    results_folder=results_folder,
                    model=model,
                    model_config={
                        "patch_size": patch_size,
                        "minimum_db": minimum_db,
                        "maximum_db": maximum_db,
                        "device": device,
                    },
                    mode=mode,
                    threshold_db=threshold_db,
                    contrast_min=contrast_min,
                    growth_allowance_db=growth_allowance_db,
                    model_threshold=float(model_threshold),
                    zero_exclusion=zero_exclusion,
                    tile_size=tile_size,
                    started=started,
                    progress=progress,
                )

            _report_progress(progress, 0.1, "Full-scene SAR physics")
            # Reuse the local water exclusion and physical criteria.
            physics_labels, physics_report = create_physics_labels(
                src,
                threshold_db=threshold_db,
                contrast_min=contrast_min,
                growth_allowance_db=growth_allowance_db,
                zero_exclusion=zero_exclusion,
            )

            evaluated = physics_labels != 255
            data = src.read(1, out_dtype="float32")

            scores = np.full(
                data.shape, -9999.0, dtype=np.float32
            )

            # Skip patches with no evaluated water pixels.
            with torch.inference_mode():
                for top in range(0, src.height, patch_size):
                    for left in range(0, src.width, patch_size):
                        height = min(patch_size, src.height - top)
                        width = min(patch_size, src.width - left)

                        tile_valid = evaluated[
                            top:top + height,
                            left:left + width,
                        ]

                        if not tile_valid.any():
                            continue

                        tile = data[
                            top:top + height,
                            left:left + width,
                        ]

                        tile = np.where(
                            np.isfinite(tile), tile, minimum_db
                        )
                        normalized = (
                            np.clip(tile, minimum_db, maximum_db)
                            - minimum_db
                        ) / (maximum_db - minimum_db)

                        padded = np.zeros(
                            (patch_size, patch_size),
                            dtype=np.float32,
                        )
                        padded[:height, :width] = normalized

                        tensor = torch.from_numpy(
                            padded
                        )[None, None].to(device)

                        prediction = torch.sigmoid(
                            model(tensor)
                        )[0, 0].cpu().numpy()[:height, :width]

                        target = scores[
                            top:top + height,
                            left:left + width,
                        ]
                        target[tile_valid] = prediction[tile_valid]

            # Both the physical criteria and model threshold must pass.
            oil = (
                (physics_labels == 1)
                & (scores >= float(model_threshold))
                & evaluated
            )

            components, _ = ndi.label(oil)
            sizes = np.bincount(components.ravel())
            keep = sizes >= 20
            keep[0] = False
            oil = keep[components]

            labels = np.full(data.shape, 255, dtype=np.uint8)
            labels[evaluated] = 0
            labels[oil] = 1

            object_ids, object_count = ndi.label(oil)
            weights = pixel_area_by_row(src)

            object_areas = None
            estimated_area = None

            if weights is not None:
                estimated_area = float(
                    np.dot(oil.sum(axis=1), weights) / 1_000_000
                )
                object_areas = np.zeros(object_count + 1)

                for row in range(src.height):
                    object_areas += (
                        np.bincount(
                            object_ids[row],
                            minlength=object_count + 1,
                        )
                        * weights[row]
                        / 1_000_000
                    )

            background = evaluated & ~oil
            mean_background = (
                float(data[background].mean())
                if background.any()
                else None
            )

            object_rows = []

            if object_count:
                ids = np.arange(1, object_count + 1)
                object_means = np.atleast_1d(
                    ndi.mean(data, object_ids, ids)
                )
                object_scores = np.atleast_1d(
                    ndi.mean(scores, object_ids, ids)
                )
                pixel_counts = np.bincount(object_ids.ravel())

                for index, object_id in enumerate(ids):
                    area = (
                        float(object_areas[object_id])
                        if object_areas is not None
                        else None
                    )
                    contrast_db = (
                        mean_background - float(object_means[index])
                        if mean_background is not None
                        else None
                    )

                    if contrast_db is None:
                        confidence = "unknown"
                    elif contrast_db >= 4 and (
                        area is None or area >= 0.1
                    ):
                        confidence = "high"
                    elif contrast_db >= 2 and (
                        area is None or area >= 0.02
                    ):
                        confidence = "medium"
                    else:
                        confidence = "low"

                    object_rows.append({
                        "object_id": int(object_id),
                        "pixel_count": int(pixel_counts[object_id]),
                        "area_km2": area,
                        "mean_backscatter_db": float(object_means[index]),
                        "contrast_vs_water_db": contrast_db,
                        "mean_model_score": float(object_scores[index]),
                        "heuristic_confidence": confidence,
                    })

            report = {
                "project": TITLE,
                "source_filename": os.path.basename(os.fspath(sar_file)),
                "model": "ProteusSegNet",
                "device": str(device),
                "mode": mode,
                "threshold_db": float(threshold_db),
                "contrast_min": float(contrast_min),
                "growth_allowance_db": float(growth_allowance_db),
                "model_threshold": float(model_threshold),
                "estimated_oil_area_km2": estimated_area,
                "selected_oil_pixels": int(oil.sum()),
                "evaluated_water_pixels": int(evaluated.sum()),
                "excluded_pixels": int((~evaluated).sum()),
                "object_summary": {
                    "num_objects": int(object_count),
                    **{
                        f"num_{level}_confidence": sum(
                            row["heuristic_confidence"] == level
                            for row in object_rows
                        )
                        for level in ["high", "medium", "low", "unknown"]
                    },
                },
                "reference_files": [
                    os.path.basename(path)
                    for path in physics_report["reference_sources"]
                ],
                "limitations": [
                    "Trained using physics-generated pseudo-labels.",
                    "Independent validation has not been completed.",
                    "Model scores are not calibrated oil probabilities.",
                    "Confidence classes are heuristic, not verified certainty.",
                ],
            }

            profile = {
                "driver": "GTiff",
                "height": src.height,
                "width": src.width,
                "count": 1,
                "crs": src.crs,
                "transform": src.transform,
                "compress": "deflate",
                "BIGTIFF": "IF_SAFER",
            }

            for filename, array, dtype, nodata in [
                ("suspected_oil_labels.tif", labels, "uint8", 255),
                ("model_scores.tif", scores, "float32", -9999.0),
                ("oil_object_ids.tif", object_ids, "int32", 0),
            ]:
                with rasterio.open(
                    os.path.join(results_folder, filename),
                    "w",
                    dtype=dtype,
                    nodata=nodata,
                    **profile,
                ) as dst:
                    dst.write(array.astype(dtype), 1)

            # Export object polygons in geographic coordinates.
            properties = {
                row["object_id"]: row for row in object_rows
            }
            features = []

            for geometry, value in shapes(
                object_ids.astype(np.int32),
                mask=oil,
                transform=src.transform,
            ):
                features.append({
                    "type": "Feature",
                    "geometry": transform_geom(
                        src.crs, "EPSG:4326", geometry
                    ),
                    "properties": properties[int(value)],
                })

            with open(
                os.path.join(results_folder, "oil_objects.geojson"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {"type": "FeatureCollection", "features": features},
                    handle,
                )

        with open(
            os.path.join(results_folder, "report.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(report, handle, indent=2)

        columns = [
            "object_id", "pixel_count", "area_km2",
            "mean_backscatter_db", "contrast_vs_water_db",
            "mean_model_score", "heuristic_confidence",
        ]

        with open(
            os.path.join(results_folder, "oil_objects.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(object_rows)

        # Figures use a downsampled preview and histogram counts, so the
        # same plotting code serves the full-scene and tiled paths.
        step = max(1, int(np.ceil(max(data.shape) / 1200)))
        values = data[evaluated]
        minimum, maximum = float(values.min()), float(values.max())
        if maximum <= minimum:
            maximum = minimum + 1
        bins = np.linspace(minimum, maximum, 61)

        graph_paths = save_analysis_figures(
            results_folder,
            preview=data[::step, ::step],
            preview_labels=labels[::step, ::step],
            display_range=np.percentile(values, [2, 98]),
            bins=bins,
            hist_background=np.histogram(data[background], bins=bins)[0],
            hist_oil=np.histogram(data[oil], bins=bins)[0],
            object_rows=object_rows,
        )

        return finish_analysis(
            work_folder, results_folder, report, graph_paths,
            processing={
                "strategy": "full scene (in memory)",
                "scene_pixels": int(data.size),
                "runtime_seconds": round(time.time() - started, 1),
            },
        )

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}", [], None

    finally:
        WORLD_COVER_FOLDER = previous_reference_folder


print("Main Proteus processing pipeline is ready.")

LATEST_DATASET_INDEX = None

# ==================================================
# 5. USER INTERFACE HELPERS
# ==================================================

MODE_DESCRIPTIONS = {
    "Conservative": "Fewer, darker and higher-contrast candidates. Use when false alarms are costly.",
    "Balanced": "Default operating point used for the GEM No.2 demonstrations.",
    "Sensitive": "Captures fainter, thinner slicks at the cost of more look-alikes.",
}

APP_CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;}
#proteus-header {background: linear-gradient(120deg,#0b1f33 0%,#102c40 55%,#14485a 100%);
  color:#fff; padding:26px 30px; border-radius:14px; border-bottom:5px solid #e8590c;}
#proteus-header h1 {color:#fff; margin:6px 0 4px; font-size:2.05rem; letter-spacing:.01em;}
#proteus-header .kicker {font-size:.78rem; letter-spacing:.18em; text-transform:uppercase; color:#a9dce2;}
#proteus-header p {margin:2px 0; color:#dbe7ee;}
.status-row {display:flex; flex-wrap:wrap; gap:8px; margin-top:14px;}
.pill {display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:999px;
  font-size:.82rem; font-weight:600; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22); color:#fff;}
.pill.ok {border-color:#4cc38a;} .pill.warn {border-color:#f0b429;} .pill.bad {border-color:#ef5350;}
.notice {padding:10px 14px; border-radius:10px; background:#fff4e6; border:1px solid #ffd8a8; color:#7a3e00; font-size:.92rem;}
.notice b {color:#b03a00;}
.kpi-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px;}
.kpi {padding:14px 16px; border-radius:12px; border:1px solid #d5dde5; background:var(--block-background-fill);}
.kpi .label {font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; opacity:.75;}
.kpi .value {font-size:1.55rem; font-weight:700; margin-top:4px;}
.kpi .sub {font-size:.8rem; opacity:.75; margin-top:2px;}
.kpi.accent {border-top:4px solid #e8590c;} .kpi.teal {border-top:4px solid #1b8a99;}
.check {padding:12px 14px; border-radius:10px; border:1px solid #d5dde5; font-size:.92rem;}
.check li {margin:3px 0;} .check ul {margin:6px 0 0 0; padding-left:4px; list-style:none;}
.ok-t {color:#2b8a3e; font-weight:700;} .warn-t {color:#c77c02; font-weight:700;} .bad-t {color:#c92a2a; font-weight:700;}
.step-title {font-weight:700; font-size:1.05rem; margin:2px 0 6px;}
footer {display:none !important;}
"""


def default_model_path():
    return str(MODEL_DIR / "proteus_model.pt")


def cached_reference_files():
    return sorted(Path(WORLD_COVER_FOLDER).glob("*.tif"))


def header_html():
    model_ok = os.path.isfile(default_model_path())
    tiles = len(cached_reference_files())
    cuda = torch.cuda.is_available()
    pills = [
        ("ok" if model_ok else "warn",
         "Model checkpoint ready" if model_ok else "No default checkpoint — upload one"),
        ("ok" if tiles else "warn",
         f"{tiles} WorldCover tile(s) cached" if tiles else "WorldCover cache empty"),
        ("ok" if cuda else "", "CUDA GPU available" if cuda else "CPU processing"),
        ("ok", f"Tiled processing up to {MAX_TILED_PIXELS / 1e9:.1f} billion px"),
    ]
    pill_html = "".join(f'<span class="pill {cls}">{html.escape(text)}</span>' for cls, text in pills)
    return f"""
<div id="proteus-header">
  <div class="kicker">Synthetic Aperture Radar · Geospatial Artificial Intelligence</div>
  <h1>Project Proteus</h1>
  <p>Mapping of Suspected Oil Slicks using SAR &amp; GeoAI — GEM No.2 case study</p>
  <p style="color:#bbd7e1">ProteusSegNet developed by Mariolino Daniel III Julabar, AlReem AlRowahi and Sultan AlShehhi</p>
  <div class="status-row">{pill_html}</div>
</div>"""


def effective_parameters_md(mode, base_threshold_db):
    threshold, contrast, growth = get_operational_params(mode, base_threshold_db)
    return (
        f"**{mode}** — {MODE_DESCRIPTIONS[mode]}  \n"
        f"Seed threshold **{threshold:.1f} dB** · minimum local contrast **{contrast:.2f}** · "
        f"region growth up to **{threshold + growth:.1f} dB**"
    )


def required_worldcover_tiles(src):
    west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    names = []
    for lat in range(int(np.floor(south / 3) * 3), int(np.floor((north - 1e-9) / 3) * 3) + 1, 3):
        for lon in range(int(np.floor(west / 3) * 3), int(np.floor((east - 1e-9) / 3) * 3) + 1, 3):
            names.append(f"ESA_WorldCover_10m_2021_v200_{worldcover_tile_name(lat, lon)}_Map.tif")
    return names, (west, south, east, north)


def inspect_sar(sar_file, reference_files=None, tile_size=DEFAULT_TILE_SIZE, strategy="Auto"):
    """Pre-flight checks shown as soon as a SAR file is uploaded."""
    if not sar_file:
        return ('<div class="check">Upload a SAR GeoTIFF to see an input check '
                '(size, CRS, dB sanity, WorldCover coverage and processing plan).</div>')
    items = []

    def add(level, text):
        icon = {"ok": "✔", "warn": "⚠", "bad": "✖"}[level]
        items.append(f'<li><span class="{level}-t">{icon}</span> {text}</li>')

    try:
        with rasterio.open(os.fspath(sar_file)) as src:
            pixels = src.width * src.height
            add("ok" if src.count == 1 else "bad",
                f"{src.count} band(s), {src.dtypes[0]} — "
                + ("single band as required." if src.count == 1 else "a single SAR band is required."))
            if src.crs is None:
                add("bad", "No CRS found. Export the SAR image as a georeferenced GeoTIFF.")
            else:
                add("ok", f"CRS {html.escape(src.crs.to_string())}, "
                          f"pixel size {abs(src.res[0]):.6g} × {abs(src.res[1]):.6g} "
                          f"({'degrees' if src.crs.is_geographic else 'map units'}).")
            add("ok", f"{src.width:,} × {src.height:,} = <b>{pixels:,}</b> pixels.")

            try:
                tiled = should_use_tiling(src.width, src.height, strategy)
                if tiled:
                    est = estimate_tiled_resources(src.width, src.height, tile_size)
                    add("ok", f"Processing plan: <b>tiled</b> — {est['tiles']} tile(s) of "
                              f"{est['tile_size']} px, ≈{est['peak_ram_mb']:,} MB peak RAM, "
                              f"≈{est['scratch_disk_mb'] + est['output_disk_mb']:,} MB temporary + output disk.")
                else:
                    add("ok", "Processing plan: <b>full scene in memory</b> (≤ 16 million pixels).")
            except ValueError as exc:
                add("bad", html.escape(str(exc)))

            # dB sanity check on a decimated read.
            scale = max(1, int(np.ceil(max(src.width, src.height) / 512)))
            sample = src.read(1, out_shape=(max(1, src.height // scale), max(1, src.width // scale)),
                              out_dtype="float32", resampling=Resampling.nearest)
            values = sample[np.isfinite(sample) & (sample != 0)]
            if values.size == 0:
                add("bad", "No finite non-zero SAR values found in a sample of the image.")
            else:
                p2, p50, p98 = np.percentile(values, [2, 50, 98])
                text = f"Backscatter sample: median {p50:.1f}, 2–98% range {p2:.1f} to {p98:.1f}."
                if p50 > 0 or (values > 0).mean() > 0.5:
                    add("bad", text + " Values look like <b>linear intensity, not dB</b>. "
                                      "Convert with 10·log10(σ⁰) before analysis.")
                elif p50 < -45 or p50 > -5:
                    add("warn", text + " Unusual for calibrated sea-surface σ⁰ in dB; check calibration.")
                else:
                    add("ok", text + " Consistent with calibrated σ⁰ in dB.")

            if src.crs is not None:
                needed, (w, s, e, n) = required_worldcover_tiles(src)
                available = {p.name for p in cached_reference_files()}
                available |= {os.path.basename(os.fspath(p)) for p in (reference_files or [])}
                missing = [name for name in needed if name not in available]
                add("ok", f"Extent: {w:.3f}°E–{e:.3f}°E, {s:.3f}°N–{n:.3f}°N.")
                if not missing:
                    add("ok", f"WorldCover coverage complete ({len(needed)} tile(s) available).")
                elif len(missing) < len(needed):
                    add("warn", "Partial WorldCover coverage; areas without a tile stay excluded. Missing: "
                                + ", ".join(html.escape(m) for m in missing))
                else:
                    add("bad", "No matching WorldCover tile available. Upload: "
                               + ", ".join(html.escape(m) for m in missing))
    except Exception as exc:
        add("bad", f"Could not read the file as a GeoTIFF: {html.escape(str(exc))}")

    return f'<div class="check"><b>Input check</b><ul>{"".join(items)}</ul></div>'


def kpi_html(report):
    if not report:
        return ('<div class="kpi-grid"><div class="kpi"><div class="label">Status</div>'
                '<div class="value">—</div><div class="sub">Run an analysis to see results.</div></div></div>')
    summary = report["object_summary"]
    area = report.get("estimated_oil_area_km2")
    water = report.get("evaluated_water_pixels") or 0
    share = 100 * report["selected_oil_pixels"] / water if water else 0
    processing = report.get("processing", {})
    strategy = processing.get("strategy", "")
    cards = [
        ("accent", "Suspected-oil area", f"{area:,.2f} km²" if area is not None else "n/a",
         f"{report['selected_oil_pixels']:,} pixels"),
        ("accent", "Slick objects", f"{summary['num_objects']:,}",
         f"High {summary['num_high_confidence']} · Medium {summary['num_medium_confidence']} · "
         f"Low {summary['num_low_confidence']}"),
        ("teal", "Share of evaluated water", f"{share:.1f} %", f"{water:,} water pixels evaluated"),
        ("teal", "Processing", "Tiled" if strategy.startswith("tiled") else "Full scene",
         f"{processing.get('runtime_seconds', 0):.1f} s · {report['device']}"),
    ]
    body = "".join(
        f'<div class="kpi {cls}"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>'
        for cls, label, value, sub in cards
    )
    return f'<div class="kpi-grid">{body}</div>'


OBJECT_TABLE_HEADERS = ["Object", "Area (km²)", "Pixels", "Mean dB", "Contrast (dB)", "Mean score", "Confidence"]


def top_objects(csv_path, limit=25):
    if not csv_path or not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def number(value, digits):
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return None

    rows.sort(key=lambda r: number(r["area_km2"], 9) or 0, reverse=True)
    return [[int(r["object_id"]), number(r["area_km2"], 4), int(r["pixel_count"]),
             number(r["mean_backscatter_db"], 2), number(r["contrast_vs_water_db"], 2),
             number(r["mean_model_score"], 3), r["heuristic_confidence"]] for r in rows[:limit]]


IN_COLAB = "google.colab" in sys.modules


def figure_value(path, title="figure"):
    """Figure for the UI. On Colab the proxy can break image links, so embed the PNG in the page."""
    if not path or not os.path.isfile(path):
        return None
    if not IN_COLAB:
        return path
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    name = os.path.basename(path)
    return (f'<div style="text-align:center"><img alt="{title}" style="max-width:100%;height:auto" '
            f'src="data:image/png;base64,{encoded}"/><br>'
            f'<a download="{name}" href="data:image/png;base64,{encoded}">⬇ Download {name}</a></div>')


def figure_component(label, **kwargs):
    if IN_COLAB:
        return gr.HTML(label=label)
    return gr.Image(label=label, type="filepath", interactive=False, show_label=False,
                    **DOWNLOAD_BUTTON, **kwargs)


def empty_results(message):
    return (message, kpi_html(None), None, None, None, [], None,
            gr.update(value=None, interactive=False), None)


def run_analysis(sar, refs, checkpoint, mode, threshold, score, zero, device,
                 strategy, tile_size, progress=gr.Progress()):
    if not sar:
        return empty_results("Please upload a calibrated single-band SAR GeoTIFF in dB.")
    try:
        progress(0, desc="Starting analysis")
        report_text, paths, archive = run_proteus_project(
            sar, refs, checkpoint, mode, threshold, score, zero, device,
            processing_strategy=strategy, tile_size=int(tile_size), progress=progress,
        )
    except Exception as exc:
        return empty_results(f"Processing error: {type(exc).__name__}: {exc}")
    if not archive:
        return empty_results(report_text)

    results = os.path.join(os.path.dirname(archive), "results")
    with open(os.path.join(results, "report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    figure = {os.path.basename(p): p for p in paths}
    files = [os.path.join(results, name) for name in
             ["oil_objects.geojson", "oil_objects.csv", "suspected_oil_labels.tif", "report.json"]
             if os.path.isfile(os.path.join(results, name))]
    return (
        report_text,
        kpi_html(report),
        figure_value(figure.get("oil_preview.png"), "Spatial delineation"),
        figure_value(figure.get("backscatter_histogram.png"), "Backscatter distribution"),
        figure_value(figure.get("object_statistics.png"), "Object statistics"),
        top_objects(os.path.join(results, "oil_objects.csv")),
        report,
        gr.update(value=archive, interactive=True),
        files,
    )


def reset_analysis_settings():
    return ("Balanced", -25, 0.5, True, "CPU", "Auto", DEFAULT_TILE_SIZE,
            effective_parameters_md("Balanced", -25))


def prepare_dataset(sar_files, refs, mode, threshold, zero, strategy, tile_size,
                    progress=gr.Progress()):
    progress(0, desc="Preparing dataset")
    message, archive = generate_dataset_from_uploads(
        sar_files, refs, mode, threshold, zero,
        processing_strategy=strategy, tile_size=int(tile_size), progress=progress,
    )
    return message, gr.update(value=archive, interactive=bool(archive))


def training_loss_figure(history):
    if not history:
        return None
    figure, axis = plt.subplots(figsize=(7, 3.6))
    try:
        axis.plot(range(1, len(history) + 1), history, marker="o", color="#1b8a99")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Weighted BCE loss")
        axis.set_title("ProteusSegNet training loss")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        path = os.path.join(str(OUTPUT_DIR), "training_loss.png")
        figure.savefig(path, dpi=130)
        return path
    finally:
        plt.close(figure)


def train_from_dataset(epochs, device, confirmed, progress=gr.Progress()):
    if not confirmed:
        return ("Tick the confirmation box to train a new model.", None,
                gr.update(value=None, interactive=False))
    if not LATEST_DATASET_INDEX or not os.path.isfile(LATEST_DATASET_INDEX):
        return ("Prepare a dataset in the Dataset Preparation tab first.", None,
                gr.update(value=None, interactive=False))
    progress(0, desc="Training ProteusSegNet")
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            path, history = train_proteus_model(LATEST_DATASET_INDEX, epochs, device)
        return (log.getvalue(), figure_value(training_loss_figure(history), "Training loss"),
                gr.update(value=path, interactive=True))
    except Exception as exc:
        return (log.getvalue() + f"\nTraining error: {type(exc).__name__}: {exc}", None,
                gr.update(value=None, interactive=False))


USER_GUIDE_MD = """
## Quick start
1. **Upload the SAR image** — a calibrated, single-band GeoTIFF in dB (e.g. `SAR_20261021_Subset-2.tif`).
   The *Input check* panel immediately reports size, CRS, a dB sanity check, WorldCover coverage and the processing plan.
2. **WorldCover and model** — cached files in `worldcover_cache` and `models/proteus_model.pt` are used automatically.
   Upload files only to add a missing tile or to test another checkpoint.
3. **Choose a sensitivity mode** — the effective thresholds are shown live beneath the selector.
4. **Execute analysis** and follow the progress bar. Large scenes are tiled automatically.
5. **Review** the KPI cards, map, histogram, object chart and the largest-object table, then download the package.

## Large scenes and tiling
Scenes above 16 million pixels are processed in tiles (default 2048 px with a 32 px halo).
Slicks that cross tile borders are merged into single objects, so results match full-scene processing.
Peak memory depends on the tile size, not on the scene size. Temporary files (about 5 bytes per pixel) are
written to the system temp folder (or `PROTEUS_SCRATCH_DIR`) and removed automatically. Choose *Always tiled* to tile small scenes too, or a
smaller tile size on low-memory machines.

## Output files
| File | Content |
|---|---|
| `suspected_oil_labels.tif` | 1 = suspected oil, 0 = evaluated water, 255 = excluded |
| `model_scores.tif` | ProteusSegNet score (0–1) on evaluated water, −9999 elsewhere |
| `oil_object_ids.tif` | Object identifier per pixel |
| `oil_objects.geojson` | Object polygons (WGS84) with attributes |
| `oil_objects.csv` | Area, backscatter, contrast, model score and heuristic confidence per object |
| `oil_preview.png`, `backscatter_histogram.png`, `object_statistics.png` | Analytical figures |
| `report.json` | Parameters, totals, processing strategy and limitations |

## Reading the results
- **Red** areas are suspected oil: darker than surrounding water, physically plausible **and** above the model score threshold.
- **Confidence classes** are heuristic (contrast and size), not calibrated probabilities.
- Low-wind areas, biogenic films and rain cells can look like oil in SAR. Treat outputs as prompts for review.
"""

METHODOLOGY_MD = """
## Processing methodology
1. Validate a georeferenced single-band SAR image expressed in dB.
2. Align local ESA WorldCover 2021 tiles; exclude non-water and unknown coverage.
3. Apply the operational backscatter threshold, local-contrast (51-px window) and region-growth criteria.
4. Run ProteusSegNet patch-wise (256 px) with the checkpoint's normalisation (−40 to +5 dB).
5. Keep detections that satisfy **both** the physics criteria and the model score threshold; remove objects under 20 pixels.
6. Export rasters, polygons, statistics, report and figures.

Scenes above 16 million pixels run through the tiled engine, which reproduces steps 2–6 tile by tile
and merges regions across tile seams.

## Limitations
- Labels are physics-generated pseudo-labels, not independently verified oil-spill annotations.
- Independent validation is incomplete; training-image predictions are demonstrations, not accuracy tests.
- Model scores and heuristic confidence classes are not calibrated oil probabilities.
- WorldCover water classes are not specifically marine classes; missing coverage remains excluded.
- Area estimates depend on CRS geometry and are not confirmed spill measurements; objects are not distinct spills.
- Tiled processing accepts up to 1.5 billion pixels and needs free disk space for temporary files.
- Figure percentiles for very large scenes are computed from the downsampled preview.
- No cloud AI APIs, pretrained weights or downloads are used during processing.

## Model identity
ProteusSegNet is a custom convolutional encoder–decoder with skip connections trained from scratch.
It is not an implementation of the published SegNet architecture.
"""


# ==================================================
# 6. APPLICATION LAYOUT
# ==================================================

# --------------------------------------------------
# Gradio 5 / 6 compatibility (Colab ships Gradio 6, the Windows app pins 5.x)
# --------------------------------------------------
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
if GRADIO_MAJOR >= 6:
    DOWNLOAD_BUTTON = {"buttons": ["download", "fullscreen"]}
    COPY_BUTTON = {"buttons": ["copy"]}
else:
    DOWNLOAD_BUTTON = {"show_download_button": True}
    COPY_BUTTON = {"show_copy_button": True}


def app_theme():
    return gr.themes.Soft(
        primary_hue="orange", secondary_hue="cyan", neutral_hue="slate",
        font=["Segoe UI", "system-ui", "sans-serif"],
    )


def style_launch_kwargs():
    """Gradio 6 takes theme and css in launch(); Gradio 5 takes them in Blocks()."""
    return {"theme": app_theme(), "css": APP_CSS} if GRADIO_MAJOR >= 6 else {}


def build_app():
    blocks_style = {} if GRADIO_MAJOR >= 6 else {"theme": app_theme(), "css": APP_CSS}
    with gr.Blocks(title=TITLE, **blocks_style) as app:
        gr.HTML(header_html())
        gr.HTML('<div class="notice">Research prototype. Outputs indicate <b>suspected</b> oil, '
                'not confirmed spills. Independent validation has not been completed.</div>')

        with gr.Tab("Analysis Workspace"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    gr.HTML('<div class="step-title">① Input data</div>')
                    sar = gr.File(label="SAR imagery — calibrated single-band GeoTIFF in dB",
                                  file_types=[".tif", ".tiff"], type="filepath")
                    check = gr.HTML(inspect_sar(None))
                    with gr.Accordion("WorldCover reference & model checkpoint (optional)", open=False):
                        refs = gr.File(label="WorldCover GeoTIFFs — keep original filenames; cached tiles are reused",
                                       file_count="multiple", file_types=[".tif", ".tiff"], type="filepath")
                        checkpoint = gr.File(label="Model checkpoint (.pt) — defaults to models/proteus_model.pt",
                                             file_types=[".pt"], type="filepath")
                with gr.Column(scale=5):
                    gr.HTML('<div class="step-title">② Detection settings</div>')
                    mode = gr.Radio(list(OPERATIONAL_MODES), value="Balanced", label="Operational sensitivity")
                    threshold = gr.Slider(-35, -10, value=-25, step=0.5, label="Base backscatter threshold (dB)",
                                          info="Seed pixels must be darker than this (after the mode offset).")
                    params = gr.Markdown(effective_parameters_md("Balanced", -25))
                    score = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Model score threshold",
                                      info="Minimum ProteusSegNet score for a pixel to be kept.")
                    with gr.Accordion("Advanced processing", open=False):
                        zero = gr.Checkbox(value=True, label="Exclude zero-valued SAR pixels (no-data)")
                        device = gr.Radio(["CPU", "Auto"], value="CPU", label="Inference device",
                                          info="Auto uses a CUDA GPU when available.")
                        strategy = gr.Radio(PROCESSING_STRATEGIES, value="Auto", label="Processing strategy",
                                            info="Auto tiles scenes above 16 million pixels.")
                        tile_size = gr.Dropdown([1024, 2048, 4096], value=DEFAULT_TILE_SIZE,
                                                label="Tile size (px)",
                                                info="Smaller tiles use less memory.")
                    with gr.Row():
                        run = gr.Button("▶ Execute analysis", variant="primary", size="lg", scale=3)
                        reset = gr.Button("Reset settings", variant="secondary", scale=1)

            gr.HTML('<div class="step-title" style="margin-top:14px">③ Results</div>')
            kpis = gr.HTML(kpi_html(None))
            with gr.Tabs():
                with gr.Tab("Map"):
                    map_image = figure_component("Figure 1 · Spatial delineation")
                with gr.Tab("Backscatter"):
                    hist_image = figure_component("Figure 2 · Backscatter distribution")
                with gr.Tab("Objects"):
                    obj_image = figure_component("Figure 3 · Object area vs contrast")
                    objects = gr.Dataframe(headers=OBJECT_TABLE_HEADERS, label="Largest 25 objects",
                                           interactive=False, wrap=True)
                with gr.Tab("Report"):
                    report = gr.Textbox(label="Processing report", lines=18, interactive=False,
                                        **COPY_BUTTON)
                    report_json = gr.JSON(label="report.json")
            gr.HTML('<div class="step-title" style="margin-top:14px">④ Deliverables</div>')
            with gr.Row():
                download = gr.DownloadButton("⬇ Download full analysis package (.zip)",
                                             interactive=False, variant="primary")
            single_files = gr.File(label="Individual outputs", file_count="multiple", interactive=False)

            for trigger in (sar.change, refs.change, tile_size.change, strategy.change):
                trigger(inspect_sar, [sar, refs, tile_size, strategy], check)
            app.load(inspect_sar, [sar, refs, tile_size, strategy], check)
            mode.change(effective_parameters_md, [mode, threshold], params)
            threshold.change(effective_parameters_md, [mode, threshold], params)
            reset.click(reset_analysis_settings, None,
                        [mode, threshold, score, zero, device, strategy, tile_size, params])
            run.click(
                run_analysis,
                [sar, refs, checkpoint, mode, threshold, score, zero, device, strategy, tile_size],
                [report, kpis, map_image, hist_image, obj_image, objects, report_json, download, single_files],
                concurrency_limit=1, concurrency_id="proteus_processing",
            )

        with gr.Tab("Dataset Preparation"):
            gr.Markdown("### Automatic pseudo-label generation\n"
                        "Optional. Converts SAR scenes into a training dataset of oil masks, validity masks, "
                        "reports and `dataset_index.csv`. Large scenes are tiled automatically.")
            with gr.Row():
                with gr.Column():
                    sar_set = gr.File(label="SAR GeoTIFFs", file_count="multiple",
                                      file_types=[".tif", ".tiff"], type="filepath")
                    refs_set = gr.File(label="WorldCover GeoTIFFs — leave empty to use the cache",
                                       file_count="multiple", file_types=[".tif", ".tiff"], type="filepath")
                with gr.Column():
                    mode_set = gr.Radio(list(OPERATIONAL_MODES), value="Balanced", label="Operational sensitivity")
                    threshold_set = gr.Slider(-35, -10, value=-25, step=0.5, label="Base threshold (dB)")
                    params_set = gr.Markdown(effective_parameters_md("Balanced", -25))
                    zero_set = gr.Checkbox(value=True, label="Exclude zero-valued SAR pixels")
                    with gr.Accordion("Advanced processing", open=False):
                        strategy_set = gr.Radio(PROCESSING_STRATEGIES, value="Auto", label="Processing strategy")
                        tile_set = gr.Dropdown([1024, 2048, 4096], value=DEFAULT_TILE_SIZE, label="Tile size (px)")
                    prepare = gr.Button("Prepare annotated dataset", variant="primary")
            prep_report = gr.Textbox(label="Preparation report", lines=12, **COPY_BUTTON)
            prep_download = gr.DownloadButton("⬇ Download training dataset (.zip)", interactive=False)
            mode_set.change(effective_parameters_md, [mode_set, threshold_set], params_set)
            threshold_set.change(effective_parameters_md, [mode_set, threshold_set], params_set)
            prepare.click(prepare_dataset,
                          [sar_set, refs_set, mode_set, threshold_set, zero_set, strategy_set, tile_set],
                          [prep_report, prep_download],
                          concurrency_limit=1, concurrency_id="proteus_processing")

        with gr.Tab("Model Training"):
            gr.Markdown("### ProteusSegNet training\n"
                        "Optional. Initialises a **new** model from the dataset prepared in this session and "
                        "overwrites `models/proteus_model.pt`. Back up existing weights first. "
                        "No validation split is created, so falling loss is not proof of accuracy.")
            with gr.Row():
                with gr.Column():
                    epochs = gr.Number(value=5, precision=0, minimum=1, label="Training epochs")
                    train_device = gr.Radio(["CPU", "Auto"], value="CPU", label="Training device")
                    confirm = gr.Checkbox(value=False,
                                          label="I authorise training a new model and overwriting the default checkpoint")
                    train = gr.Button("Start model training", variant="primary")
                with gr.Column():
                    loss_plot = (gr.HTML(label="Training loss") if IN_COLAB
                                 else gr.Image(label="Training loss", type="filepath", interactive=False))
            train_report = gr.Textbox(label="Training report", lines=12, **COPY_BUTTON)
            saved_model = default_model_path()
            model_download = gr.DownloadButton(
                "⬇ Download proteus_model.pt",
                value=saved_model if os.path.isfile(saved_model) else None,
                interactive=os.path.isfile(saved_model))
            train.click(train_from_dataset, [epochs, train_device, confirm],
                        [train_report, loss_plot, model_download],
                        concurrency_limit=1, concurrency_id="proteus_processing")

        with gr.Tab("User Guide"):
            gr.Markdown(USER_GUIDE_MD)

        with gr.Tab("Methodology & Limitations"):
            gr.Markdown(METHODOLOGY_MD)

        gr.Markdown("<center><small>Project Proteus · National Space Academy 2026 — EO Application "
                    "Development (Level 2) · Research and educational prototype · "
                    "All detections represent suspected oil.</small></center>")
    return app


def load_credentials():
    """Optional sign-in: environment variables override login.json."""
    username = os.environ.get("PROTEUS_USERNAME")
    password = os.environ.get("PROTEUS_PASSWORD")
    login_file = BASE_DIR / "login.json"
    if not (username and password) and login_file.is_file():
        with open(login_file, encoding="utf-8") as handle:
            data = json.load(handle)
        username = username or data.get("username")
        password = password or data.get("password")
    return (username, password) if username and password else None


def main():
    app = build_app()
    app.queue(default_concurrency_limit=1)
    credentials = load_credentials()
    if credentials is None:
        print("No login.json or PROTEUS_USERNAME/PROTEUS_PASSWORD found: sign-in disabled "
              "(the server listens on 127.0.0.1 only).")
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PROTEUS_PORT", 7860)),
        share=False,
        inbrowser=True,
        auth=credentials,
        auth_message=("<div style='text-align:center'><h2>Project Proteus</h2>"
                      "<p>ProteusSegNet developed by Mariolino Daniel III Julabar, AlReem AlRowahi and Sultan AlShehhi</p></div>"),
        allowed_paths=[str(OUTPUT_DIR), str(MODEL_DIR)],
        **style_launch_kwargs(),
    )


if __name__ == "__main__":
    main()
