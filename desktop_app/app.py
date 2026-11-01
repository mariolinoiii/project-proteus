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


def run_proteus_project(
    sar_file,
    reference_files,
    model_file,
    mode,
    base_threshold_db,
    model_threshold,
    zero_exclusion,
    device_choice,
):
    """
    Run the integrated Proteus pipeline.

    Final detections must satisfy:
    1. Valid SAR and local reference-water coverage.
    2. SAR physics pseudo-label criteria.
    3. The trained model's score threshold.

    Model scores and object confidence classes are not calibrated
    probabilities of a real oil spill.
    """
    global WORLD_COVER_FOLDER, LATEST_PROTEUS_ZIP

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

        # Graph 1: downsampled map preview.
        step = max(1, int(np.ceil(max(data.shape) / 1200)))
        preview = data[::step, ::step]
        preview_labels = labels[::step, ::step]
        preview_valid = preview_labels != 255

        values = data[evaluated]
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1

        overlay = np.zeros((*preview.shape, 4), dtype=np.float32)
        overlay[preview_labels == 1] = [1, 0, 0, 0.65]

        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        try:
            for axis in axes:
                axis.imshow(
                    np.ma.array(preview, mask=~preview_valid),
                    cmap="gray",
                    vmin=low,
                    vmax=high,
                )
                axis.axis("off")

            axes[0].set_title("Evaluated reference-water area")
            axes[1].imshow(overlay)
            axes[1].set_title("Integrated suspected-oil delineation")
            figure.tight_layout()

            path = os.path.join(results_folder, "oil_preview.png")
            figure.savefig(path, dpi=150)
            graph_paths.append(path)
        finally:
            plt.close(figure)

        # Graph 2: backscatter distributions.
        figure, axis = plt.subplots(figsize=(9, 5))
        try:
            minimum, maximum = float(values.min()), float(values.max())
            if maximum <= minimum:
                maximum = minimum + 1
            bins = np.linspace(minimum, maximum, 61)

            for selection, name, color in [
                (background, "Evaluated non-oil water", "tab:blue"),
                (oil, "Suspected oil", "tab:red"),
            ]:
                if selection.any():
                    axis.hist(
                        data[selection],
                        bins=bins,
                        density=True,
                        alpha=0.5,
                        label=name,
                        color=color,
                    )

            axis.set_xlabel("SAR backscatter (dB)")
            axis.set_ylabel("Probability density")
            axis.set_title("Backscatter distribution")
            axis.legend()
            figure.tight_layout()

            path = os.path.join(results_folder, "backscatter_histogram.png")
            figure.savefig(path, dpi=150)
            graph_paths.append(path)
        finally:
            plt.close(figure)

        # Graph 3: object area versus physical contrast.
        figure, axis = plt.subplots(figsize=(9, 5))
        try:
            plotted = False

            for level, color in [
                ("high", "green"),
                ("medium", "orange"),
                ("low", "red"),
                ("unknown", "gray"),
            ]:
                rows = [
                    row for row in object_rows
                    if row["heuristic_confidence"] == level
                    and row["area_km2"] is not None
                    and row["contrast_vs_water_db"] is not None
                    and row["area_km2"] > 0
                ]

                if rows:
                    axis.scatter(
                        [row["area_km2"] for row in rows],
                        [row["contrast_vs_water_db"] for row in rows],
                        label=level,
                        color=color,
                    )
                    plotted = True

            if plotted:
                axis.set_xscale("log")
                axis.legend(title="Heuristic confidence")
            else:
                axis.text(
                    0.5, 0.5,
                    "No objects with available area and contrast.",
                    transform=axis.transAxes,
                    ha="center",
                )

            axis.set_xlabel("Object area (km²)")
            axis.set_ylabel("Background-water mean − object mean (dB)")
            axis.set_title("Object area versus backscatter contrast")
            figure.tight_layout()

            path = os.path.join(results_folder, "object_statistics.png")
            figure.savefig(path, dpi=150)
            graph_paths.append(path)
        finally:
            plt.close(figure)

        zip_path = shutil.make_archive(
            os.path.join(work_folder, "Proteus_results"),
            "zip",
            results_folder,
        )
        LATEST_PROTEUS_ZIP = zip_path

        area_text = (
            f"{estimated_area:.4f} km²"
            if estimated_area is not None
            else "Unavailable for this CRS geometry"
        )

        summary = (
            f"Project Proteus | Analysis completed\n\n"
            f"Source: {report['source_filename']}\n"
            f"Device: {device}\n"
            f"Mode: {mode}\n"
            f"Effective SAR threshold: {threshold_db:.2f} dB\n"
            f"Model score threshold: {float(model_threshold):.2f}\n"
            f"Suspected-oil pixels: {int(oil.sum()):,}\n"
            f"Estimated suspected-oil area: {area_text}\n"
            f"Objects: {object_count}\n"
            f"Excluded pixels: {int((~evaluated).sum()):,}\n\n"
            "Final detections passed the trained model AND SAR physics rules.\n"
            "Missing reference coverage remains excluded.\n"
            "Independent validation has not been completed.\n\n"
            "Download the analysis package using the button below the figures."
        )

        return summary, graph_paths, zip_path

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}", [], None

    finally:
        WORLD_COVER_FOLDER = previous_reference_folder


print("Main Proteus processing pipeline is ready.")

LATEST_DATASET_INDEX = None

def inline_figures(paths):
    titles = {"oil_preview.png":"Figure 1. Spatial delineation",
              "backscatter_histogram.png":"Figure 2. Backscatter distribution",
              "object_statistics.png":"Figure 3. Object characteristics"}
    parts = []
    for path in paths:
        with open(path,"rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        caption = html.escape(titles.get(os.path.basename(path), os.path.basename(path)))
        parts.append(f'<section style="margin:0 0 24px;padding:16px;border:1px solid #8996a5;border-radius:8px;">'
                     f'<h3>{caption}</h3><img src="data:image/png;base64,{encoded}" '
                     f'alt="{caption}" style="width:100%;height:auto;background:white"></section>')
    return "\n".join(parts)


def run_analysis(*args):
    try:
        if args[0]:
            with rasterio.open(os.fspath(args[0])) as image:
                if image.width * image.height > 16_000_000:
                    return "Input exceeds 16,000,000 pixels; automatic tiling is not implemented.", "", gr.update(value=None, interactive=False)
        report, paths, archive = run_proteus_project(*args)
        return report, inline_figures(paths), gr.update(value=archive, interactive=bool(archive))
    except Exception as exc:
        return f"Processing error: {type(exc).__name__}: {exc}", "", gr.update(value=None, interactive=False)


def prepare_dataset(*args):
    message, archive = generate_dataset_from_uploads(*args)
    return message, gr.update(value=archive, interactive=bool(archive))


def train_from_dataset(epochs, device, confirmed):
    if not confirmed:
        return "Explicit training confirmation is required.", gr.update(value=None, interactive=False)
    if not LATEST_DATASET_INDEX or not os.path.isfile(LATEST_DATASET_INDEX):
        return "Prepare a dataset in the Dataset Preparation tab first.", gr.update(value=None, interactive=False)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            path, history = train_proteus_model(LATEST_DATASET_INDEX, epochs, device)
        return log.getvalue(), gr.update(value=path, interactive=True)
    except Exception as exc:
        return log.getvalue() + f"\nTraining error: {exc}", gr.update(value=None, interactive=False)


def build_app():
    with gr.Blocks(title=TITLE) as app:
        gr.HTML("""<header style="background:#102c40;color:white;padding:24px;border-radius:10px;border-bottom:5px solid #238793;">
<h1 style="color:white;margin:0 0 8px;">Project Proteus</h1>
<p>Mapping of Suspected Oil Slicks using SAR &amp; GeoAI</p>
<p>Developed by Mariolino Daniel III Julabar y Gonzales</p>
<p>ProteusSegNet · Local water-reference masking · SAR physics filtering</p></header>""")
        gr.Markdown("Research prototype. All detections indicate suspected oil. Independent validation is incomplete.")
        with gr.Tab("Analysis Workspace"):
            gr.Markdown("## 1. Input data and configuration")
            with gr.Row():
                with gr.Column():
                    sar = gr.File(label="Calibrated single-band SAR GeoTIFF in dB", file_types=[".tif",".tiff"], type="filepath")
                    refs = gr.File(label="Optional WorldCover uploads — cached files are reused", file_count="multiple", file_types=[".tif",".tiff"], type="filepath")
                    checkpoint = gr.File(label="Optional model upload — defaults to models/proteus_model.pt", file_types=[".pt"], type="filepath")
                with gr.Column():
                    mode = gr.Radio(list(OPERATIONAL_MODES), value="Balanced", label="Operational sensitivity")
                    threshold = gr.Slider(-35,-10,value=-25,step=0.5,label="Base backscatter threshold (dB)")
                    score = gr.Slider(0.05,0.95,value=0.5,step=0.05,label="Model score threshold")
                    zero = gr.Checkbox(value=True,label="Exclude zero-valued SAR pixels")
                    device = gr.Radio(["CPU","Auto"],value="CPU",label="Inference device")
                    run = gr.Button("Execute analysis",variant="primary")
            gr.Markdown("## 2. Analysis summary")
            report = gr.Textbox(label="Processing report",lines=16,interactive=False)
            gr.Markdown("## 3. Maps and analytical figures")
            figures = gr.HTML()
            gr.Markdown("## 4. Download analysis deliverables")
            download = gr.DownloadButton(label="Download analysis package (.zip)",interactive=False)
            run.click(run_analysis,[sar,refs,checkpoint,mode,threshold,score,zero,device],
                      [report,figures,download],concurrency_limit=1,concurrency_id="proteus_processing")
        with gr.Tab("Dataset Preparation"):
            gr.Markdown("## Automatic pseudo-label generation")
            sar_set = gr.File(label="SAR GeoTIFFs",file_count="multiple",file_types=[".tif",".tiff"],type="filepath")
            refs_set = gr.File(label="Optional WorldCover files — leave empty to use cache",file_count="multiple",file_types=[".tif",".tiff"],type="filepath")
            mode_set = gr.Radio(list(OPERATIONAL_MODES),value="Balanced",label="Operational sensitivity")
            threshold_set = gr.Slider(-35,-10,value=-25,step=0.5,label="Base threshold (dB)")
            zero_set = gr.Checkbox(value=True,label="Exclude zero-valued SAR pixels")
            prepare = gr.Button("Prepare annotated dataset",variant="primary")
            prep_report = gr.Textbox(label="Preparation report",lines=12)
            prep_download = gr.DownloadButton(label="Download training dataset (.zip)",interactive=False)
            prepare.click(prepare_dataset,[sar_set,refs_set,mode_set,threshold_set,zero_set],
                          [prep_report,prep_download],concurrency_limit=1,concurrency_id="proteus_processing")
        with gr.Tab("Model Training"):
            gr.Markdown("## ProteusSegNet training")
            gr.Markdown("Optional: initializes a NEW model using the dataset prepared in this session. "
                        "Overwrites models/proteus_model.pt. Save existing weights first. No validation split is created.")
            epochs = gr.Number(value=5,precision=0,minimum=1,label="Training epochs")
            train_device = gr.Radio(["CPU","Auto"],value="CPU",label="Training device")
            confirm = gr.Checkbox(value=False,label="I authorize training a new model and overwriting the default checkpoint")
            train = gr.Button("Start model training",variant="primary")
            train_report = gr.Textbox(label="Training report — appears upon completion",lines=12)
            saved_model = str(MODEL_DIR / "proteus_model.pt")
            model_download = gr.DownloadButton(label="Download proteus_model.pt",value=saved_model if os.path.isfile(saved_model) else None,interactive=os.path.isfile(saved_model))
            train.click(train_from_dataset,[epochs,train_device,confirm],[train_report,model_download],
                        concurrency_limit=1,concurrency_id="proteus_processing")
        with gr.Tab("Methodology & Limitations"):
            gr.Markdown("""
## Methodology
ProteusSegNet predictions are combined with local WorldCover water exclusion,
backscatter thresholds, local contrast and connected-region filtering.
The custom encoder–decoder uses skip connections; it is not the published SegNet architecture.

## Limitations
- Maximum 16 million pixels per input; automatic tiling and merged outputs are not implemented.
- Physics pseudo-labels are not independently verified oil annotations.
- Independent validation and comparison against a physics-only baseline are outstanding.
- Training-image predictions are demonstrations, not accuracy tests.
- Model scores and heuristic confidence classes are not calibrated oil probabilities.
- WorldCover water is not specifically marine water; unknown coverage is excluded.
- Area estimates are CRS-dependent approximations. Object count is not a count of distinct spills.
- CPU processing, polygonization and full-scene arrays can require substantial time and RAM.
- Local reference cache only: this version does not download missing WorldCover tiles.
- Normal startup uses no cloud AI API and does not retrain or download pretrained models.

## Local storage
Outputs are retained in the outputs folder. Remove old runs manually when no longer needed.
""")
    return app


def main():
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        allowed_paths=[str(OUTPUT_DIR), str(MODEL_DIR)]
    )


if __name__ == "__main__":
    main()