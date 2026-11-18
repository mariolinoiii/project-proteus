"""Application services for Project Proteus / ProteusSegNet (standard library HTTP; no web framework)."""
import os, sys, io, re, csv, json, time, uuid, shutil, threading, traceback, subprocess, contextlib, datetime as dt
import urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("PROTEUS_HOME", ROOT / "workspace")).resolve()
os.environ["PROTEUS_HOME"] = str(HOME)
UPLOADS, CACHE, MODELS, OUT = HOME / "uploads", HOME / "worldcover_cache", HOME / "proteus_models", HOME / "proteus_outputs"
for d in (UPLOADS, CACHE, MODELS, OUT):
    d.mkdir(parents=True, exist_ok=True)
LATEST_FILE, ERROR_LOG = HOME / "latest_dataset.json", HOME / "error.log"
MODEL_PATH = MODELS / "proteus_model.pt"
WC_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
BUILD = "v3"
SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]")
MAX_UPLOAD = 20 * 1024 ** 3

_core, _core_error = None, None


def core():
    global _core, _core_error
    if _core is None and _core_error is None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import proteus_core
            _core = proteus_core
        except Exception as exc:
            _core_error = f"{type(exc).__name__}: {exc}"
    if _core is None:
        raise RuntimeError("Scientific core unavailable (" + _core_error + "). Run Start_Proteus.bat to prepare dependencies.")
    return _core


def log_error(text):
    try:
        with open(ERROR_LOG,"a",encoding="utf-8") as h:
            h.write(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {text}\n")
    except OSError:
        try:
            print(str(text),file=sys.__stderr__,flush=True)
        except OSError:
            pass


def clean_name(name):
    name = SAFE_NAME.sub("_", os.path.basename(name or "")).strip(". ")
    if not name:
        raise ValueError("Invalid file name.")
    return name


# ------------------------------------------------------------------ jobs
JOBS, JOB_LOCK, ACTIVE = {}, threading.Lock(), [None]


def job_log(job, text):
    text = str(text).strip()[:300]
    if text and (not job["log"] or job["log"][-1] != text):
        job["log"].append(text)
        del job["log"][:-300]


class Tee(io.TextIOBase):
    def __init__(self, job):
        self.job, self.buf = job, ""

    def write(self, text):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            job_log(self.job, line)
            m = re.match(r"Epoch (\d+)/(\d+)", line.strip())
            if m:
                self.job["progress"] = 0.1 + 0.88 * int(m.group(1)) / int(m.group(2))
                self.job["message"] = line.strip()[:120]
        self.job["heartbeat"] = time.time()
        return len(text)


def start_job(kind, fn, **kwargs):
    with JOB_LOCK:
        cur = ACTIVE[0] and JOBS.get(ACTIVE[0])
        if cur and cur["status"] == "running":
            raise RuntimeError("Another job is running. Wait for it to finish.")
        jid = uuid.uuid4().hex[:10]
        job = JOBS[jid] = {"id": jid, "kind": kind, "status": "running", "progress": 0.0, "message": "Starting", "device": None,
                           "log": [], "result": None, "error": None, "started": time.time(), "heartbeat": time.time(), "elapsed": 0}
        ACTIVE[0] = jid

    def runner():
        try:
            with contextlib.redirect_stdout(Tee(job)):
                job["result"] = fn(job, **kwargs)
            job["status"], job["progress"], job["message"] = "done", 1.0, "Complete"
            job_log(job, "Complete.")
        except Exception as exc:
            log_error(f"{kind}: {traceback.format_exc()}")
            job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
            job_log(job, "ERROR: " + job["error"])
        job["elapsed"] = round(time.time() - job["started"], 1)

    threading.Thread(target=runner, daemon=True).start()
    return jid


def progress_cb(job, lo=0.0, hi=1.0):
    def cb(fraction, desc=""):
        job["progress"] = lo + (hi - lo) * float(fraction)
        job["heartbeat"] = time.time()
        if desc:
            job["message"] = str(desc)
            job_log(job, desc)
    return cb


def get_job(jid):
    job = JOBS.get(jid)
    if job and job["status"] == "running":
        job["elapsed"] = round(time.time() - job["started"], 1)
    return job


def active_job():
    jid = ACTIVE[0]
    return JOBS[jid] if jid and JOBS[jid]["status"] == "running" else None


# ------------------------------------------------------------------ diagnostics / state
def nvidia_driver():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=6)
        return out.stdout.strip() or None
    except Exception:
        return None


def diagnostics():
    info = {"python": sys.version.split()[0], "torch": None, "cuda_build": None, "cuda_ok": False, "gpu": None,
            "driver": nvidia_driver(), "note": "", "core_error": None}
    try:
        c = core()
        info["torch"], info["cuda_build"] = c.torch.__version__, c.torch.version.cuda
        ok, why = c.cuda_works()
        info["cuda_ok"], info["note"] = ok, ("GPU ready" if ok else why)
        if ok:
            info["gpu"] = c.torch.cuda.get_device_name(0)
    except Exception as exc:
        info["core_error"] = str(exc)
    info["auto_resolves_to"] = "GPU" if info["cuda_ok"] else "CPU"
    return info


def list_files(folder, exts):
    return sorted((p.name for p in folder.iterdir() if p.suffix.lower() in exts), key=str.lower)


def load_latest_dataset():
    try:
        data = json.loads(LATEST_FILE.read_text(encoding="utf-8"))
        return data if Path(data["index_path"]).is_file() else None
    except Exception:
        return None


def checkpoint_info():
    if not MODEL_PATH.is_file():
        return {"ready": False}
    return {"ready": True, "modified": dt.datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(timespec="minutes"),
            "size_mb": round(MODEL_PATH.stat().st_size / 1e6, 2)}


def state():
    j = active_job()
    return {"build": BUILD, "home": str(HOME), "checkpoint": checkpoint_info(),
            "uploads": {"sar": list_files(UPLOADS, {".tif", ".tiff"}), "models": list_files(UPLOADS, {".pt"})},
            "worldcover_cache": list_files(CACHE, {".tif", ".tiff"}), "dataset": load_latest_dataset(), "job": j and j["id"]}


def save_upload(kind, name, stream, length):
    name = clean_name(name)
    ext = Path(name).suffix.lower()
    if kind in ("sar", "reference") and ext not in (".tif", ".tiff"):
        raise ValueError("Only .tif / .tiff files are accepted.")
    if kind == "model" and ext != ".pt":
        raise ValueError("Checkpoints must be .pt files.")
    if kind not in ("sar", "reference", "model"):
        raise ValueError("Unknown upload type.")
    if length <= 0 or length > MAX_UPLOAD:
        raise ValueError("Invalid upload size.")
    folder = CACHE if kind == "reference" else UPLOADS
    if shutil.disk_usage(folder).free < length * 1.1:
        raise IOError("Not enough free disk space for this upload.")
    tmp = folder / (name + ".part")
    remaining = length
    with open(tmp, "wb") as h:
        while remaining > 0:
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.write(chunk)
            remaining -= len(chunk)
    if remaining:
        tmp.unlink(missing_ok=True)
        raise IOError("Upload interrupted.")
    os.replace(tmp, folder / name)
    return name


# ------------------------------------------------------------------ WorldCover
def tile_filename(lat, lon):
    return f"ESA_WorldCover_10m_2021_v200_{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}_Map.tif"


def tiles_for_bounds(west, south, east, north):
    import math
    return [tile_filename(la, lo) for la in range(int(math.floor(south / 3) * 3), int(math.floor((north - 1e-9) / 3) * 3) + 1, 3)
            for lo in range(int(math.floor(west / 3) * 3), int(math.floor((east - 1e-9) / 3) * 3) + 1, 3)]


def sar_bounds(path):
    c = core()
    with c.rasterio.open(path) as s:
        if s.crs is None:
            raise ValueError("The SAR image must have a CRS.")
        return c.transform_bounds(s.crs, "EPSG:4326", *s.bounds, densify_pts=21)


def acquisition_time(src):
    try:
        for k, v in (src.tags() or {}).items():
            if any(t in k.upper() for t in ("ACQUISITION", "START_TIME", "FIRST_LINE", "DATETIME")) and v:
                return str(v)
    except Exception:
        pass
    return None


def inspect_sar(name):
    import numpy as np
    c = core()
    path = UPLOADS / clean_name(name)
    if not path.is_file():
        raise FileNotFoundError("SAR file not found in uploads.")
    with c.rasterio.open(path) as s:
        issues, notes = [], []
        step = max(1, int(np.ceil(max(s.width, s.height) / 1024)))
        arr = s.read(1, out_shape=(max(1, s.height // step), max(1, s.width // step)))
        vals = arr[np.isfinite(arr)]
        if s.nodata is not None:
            vals = vals[vals != s.nodata]
        stats = None
        if vals.size:
            p = np.percentile(vals, [1, 50, 99])
            stats = {"p1": float(p[0]), "median": float(p[1]), "p99": float(p[2])}
            if p[2] > 40 or vals.min() >= 0:
                issues.append("Values look like linear backscatter, not dB. Convert before use.")
            if p[0] < -80:
                notes.append("Very low dB values detected; zero exclusion is recommended.")
        else:
            issues.append("No finite sample values found.")
        if s.count != 1:
            issues.append(f"Expected exactly one band; found {s.count}.")
        bounds, tiles, cached = None, [], []
        if s.crs is None:
            issues.append("No CRS defined.")
        else:
            w, so, e, n = c.transform_bounds(s.crs, "EPSG:4326", *s.bounds, densify_pts=21)
            bounds = [w, so, e, n]
            if e - w > 12 or n - so > 12 or not (-180 <= w < e <= 180):
                issues.append("Scene exceeds 12 degrees or crosses the antimeridian.")
            else:
                tiles = tiles_for_bounds(w, so, e, n)
                cached = [t for t in tiles if (CACHE / t).is_file()]
        pixels = s.width * s.height
        plan = "Full scene in memory" if pixels <= 16_000_000 else ("Tiled processing" if pixels <= 1_500_000_000 else "Exceeds configured tiled limit")
        if pixels > 1_500_000_000:
            issues.append("Scene exceeds the configured tiled limit (~1.5 billion pixels).")
        if pixels > 16_000_000:
            notes.append("Large scene: processing can take tens of minutes or longer.")
        return {"name": path.name, "width": s.width, "height": s.height, "bands": s.count, "crs": str(s.crs) if s.crs else None,
                "bounds": bounds, "stats": stats, "tiles": tiles, "cached_tiles": cached, "missing_tiles": [t for t in tiles if t not in cached],
                "plan": plan, "issues": issues, "notes": notes, "acquisition": acquisition_time(s)}


def download_tile(filename, job=None, lo=0.0, hi=1.0):
    """Resumable download. Returns 'cached' | 'downloaded' | 'unavailable'."""
    dest, part = CACHE / filename, CACHE / (filename + ".part")
    if dest.is_file():
        return "cached"
    for attempt in range(3):
        have = part.stat().st_size if part.is_file() else 0
        req = urllib.request.Request(WC_PREFIX + filename, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resumed = r.status == 206
                total = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
                if total and shutil.disk_usage(CACHE).free < total * 1.1:
                    raise IOError("Not enough free disk space for this WorldCover tile.")
                done = have if resumed else 0
                with open(part, "ab" if resumed else "wb") as h:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        h.write(chunk)
                        done += len(chunk)
                        if job is not None and total:
                            job["progress"] = lo + (hi - lo) * done / total
                            job["heartbeat"] = time.time()
                if total and done < total:
                    raise IOError("Incomplete download")
            core().rasterio.open(part).close()
            os.replace(part, dest)
            return "downloaded"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                part.unlink(missing_ok=True)
                return "unavailable"
            if exc.code == 416:
                part.unlink(missing_ok=True)
            if attempt == 2:
                raise
        except Exception as exc:
            if "disk space" in str(exc) or attempt == 2:
                raise
            time.sleep(2)
    return "unavailable"


def ensure_references(job, scene_names, mode, lo=0.0, hi=0.1):
    needed = []
    for n in scene_names:
        for t in tiles_for_bounds(*sar_bounds(UPLOADS / clean_name(n))):
            if t not in needed:
                needed.append(t)
    rep = {"needed": needed, "downloaded": [], "cached": [], "unavailable": []}
    for i, t in enumerate(needed):
        a, b = lo + (hi - lo) * i / max(1, len(needed)), lo + (hi - lo) * (i + 1) / len(needed)
        if mode == "online":
            msg = f"WorldCover tile {i + 1}/{len(needed)}: {t}"
            job["message"] = msg
            job_log(job, msg)
            st = download_tile(t, job, a, b)
        else:
            st = "cached" if (CACHE / t).is_file() else "unavailable"
        rep["downloaded" if st == "downloaded" else "cached" if st == "cached" else "unavailable"].append(t)
    if not rep["cached"] and not rep["downloaded"]:
        raise RuntimeError("No WorldCover reference is available for this scene. " +
                           ("The provider has no tile for the area." if mode == "online" else
                            "Upload the required tiles or switch to Online: " + ", ".join(needed)))
    job_log(job, f"References: {len(rep['cached'])} cached, {len(rep['downloaded'])} downloaded, {len(rep['unavailable'])} unavailable (excluded).")
    return rep


# ------------------------------------------------------------------ analysis post-processing
def log_norm(v, lo, hi):
    import math
    return min(1.0, max(0.0, (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))))


def add_confidence_index(csv_path):
    """Additive confidence_index/class columns; legacy heuristic_confidence is untouched."""
    with open(csv_path, newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))

    def num(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError, KeyError):
            return None
    for r in rows:
        a, c, d, m = (num(r, k) for k in ("area_km2", "contrast_vs_water_db", "mean_backscatter_db", "mean_model_score"))
        parts = []
        if a and a > 0:
            parts.append((0.30, log_norm(a, 0.005, 0.5)))
        if c is not None:
            parts.append((0.40, min(1.0, max(0.0, c / 6.0))))
        if d is not None:
            parts.append((0.10, min(1.0, max(0.0, (-15.0 - d) / 20.0))))
        if m is not None:
            parts.append((0.20, min(1.0, max(0.0, m))))
        if c is None:
            r["confidence_index"], r["confidence_class"] = "", "unknown"
        else:
            idx = 100 * sum(w * v for w, v in parts) / sum(w for w, _ in parts)
            r["confidence_index"], r["confidence_class"] = f"{idx:.1f}", ("high" if idx >= 65 else "medium" if idx >= 35 else "low")
    cols = ["object_id", "pixel_count", "area_km2", "mean_backscatter_db", "contrast_vs_water_db", "mean_model_score",
            "heuristic_confidence", "confidence_index", "confidence_class"]
    with open(csv_path, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def screening(report):
    summary = report["object_summary"]
    area = report.get("estimated_oil_area_km2") or 0.0
    ev = report.get("evaluated_water_pixels") or 0
    frac = report["selected_oil_pixels"] / ev if ev else 0.0
    flags = []
    if area >= 5:
        flags.append("Suspected area >= 5 km²")
    if summary["num_high_confidence"] >= 3:
        flags.append(">= 3 legacy high-confidence objects")
    if frac >= 0.01:
        flags.append("Affected evaluated water >= 1%")
    if summary["num_objects"] == 0:
        sev = "None detected"
    elif area >= 5 or len(flags) >= 2:
        sev = "High"
    elif area >= 0.5 or flags or summary["num_high_confidence"]:
        sev = "Moderate"
    else:
        sev = "Low"
    return {"severity": sev, "flags": flags, "affected_water_pct": round(100 * frac, 4)}


def to_cog(path):
    """Rewrite a raster as a Cloud Optimized GeoTIFF in place. CRS, transform, size and nodata are preserved."""
    import rasterio
    from rasterio.shutil import copy as rcopy
    tmp = Path(str(path) + ".cog.tmp")
    try:
        rcopy(str(path), str(tmp), driver="COG", compress="DEFLATE", overview_resampling="NEAREST", blocksize=256, bigtiff="IF_SAFER")
    except Exception:
        with rasterio.open(path) as s:
            prof = s.profile
            prof.update(driver="GTiff", tiled=True, blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER")
            with rasterio.open(tmp, "w", **prof) as d:
                for _, win in s.block_windows(1):
                    d.write(s.read(1, window=win), 1, window=win)
        with rasterio.open(tmp, "r+") as d:
            d.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.nearest)
    os.replace(tmp, path)


def write_overlay_rgb(sar_path, labels_path, out_path, display_range):
    """8-bit RGB GeoTIFF (SAR greyscale, suspected oil amber, excluded dark) in the SAR's native CRS and grid."""
    import numpy as np
    import rasterio
    lo, hi = display_range
    with rasterio.open(sar_path) as s, rasterio.open(labels_path) as lab:
        prof = s.profile
        prof.update(driver="GTiff", count=3, dtype="uint8", nodata=None, tiled=True, blockxsize=256, blockysize=256,
                    compress="deflate", photometric="RGB", BIGTIFF="IF_SAFER")
        with rasterio.open(out_path, "w", **prof) as d:
            for _, win in s.block_windows(1):
                v = s.read(1, window=win).astype("float32")
                l = lab.read(1, window=win)
                g = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
                g8 = (np.where(np.isfinite(g), g, 0) * 255).astype("uint8")
                r, gr, b = g8.copy(), g8.copy(), g8.copy()
                m, x = l == 1, l == 255
                r[m], gr[m], b[m] = 232, 137, 12
                r[x], gr[x], b[x] = 14, 22, 36
                d.write(np.stack([r, gr, b]), window=win)


def enrich_geojson(results, rows, run_id, scene):
    gj = results / "oil_objects.geojson"
    props = {str(r["object_id"]): r for r in rows}
    problems = []
    if gj.is_file():
        data = json.loads(gj.read_text(encoding="utf-8"))
        keep = []
        for f in data.get("features", []):
            g = f.get("geometry")
            if not g or not g.get("coordinates"):
                problems.append("empty geometry removed")
                continue
            p = f.setdefault("properties", {})
            row = props.get(str(p.get("object_id")), {})
            p.update({"status": "suspected", "confidence_index": row.get("confidence_index", ""),
                      "confidence_class": row.get("confidence_class", ""), "scene": scene, "run_id": run_id})
            keep.append(f)
        data["features"] = keep
        data["name"] = f"Proteus suspected oil objects - {scene}"
        gj.write_text(json.dumps(data), encoding="utf-8")
    return problems


def write_footprint(results, bounds, scene, run_id):
    w, s, e, n = bounds
    feat = {"type": "Feature", "properties": {"scene": scene, "run_id": run_id, "role": "scene footprint (WGS84 bounding box)"},
            "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}
    (results / "scene_footprint.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feat]}), encoding="utf-8")


GIS_README = """Project Proteus - GIS-ready outputs (suspected oil, not confirmed spills)

RASTERS (native CRS, transform and grid of the input SAR scene; Cloud Optimized GeoTIFF)
  suspected_oil_labels.tif   1 = suspected oil, 0 = evaluated water, 255 = excluded (nodata)
  model_scores.tif           float32 model score; -9999 outside evaluation (nodata). Heuristic, not a probability
  oil_object_ids.tif         int32 object identifier; 0 = background (nodata)
  oil_overlay_rgb.tif        8-bit RGB: SAR greyscale, suspected oil amber, excluded dark. Display product only

VECTORS (WGS84 / EPSG:4326, as GeoJSON requires)
  oil_objects.geojson        one polygon feature per object; properties include area, contrast, scores, confidence
  scene_footprint.geojson    WGS84 bounding box of the scene (area of interest)

Suggested upload types in gIQ (verify against current gIQ documentation): rasters as SAR, vector as Polygons.
Outputs are screening products for analyst review. Independent validation has not been completed.
"""


def validate_outputs(results):
    import rasterio
    msgs = []
    for n in ("suspected_oil_labels.tif", "model_scores.tif", "oil_object_ids.tif", "oil_overlay_rgb.tif"):
        p = results / n
        if p.is_file():
            with rasterio.open(p) as s:
                if s.crs is None:
                    msgs.append(f"{n}: missing CRS")
    for n in ("oil_objects.geojson", "scene_footprint.geojson"):
        p = results / n
        if p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("type") != "FeatureCollection":
            msgs.append(f"{n}: not a FeatureCollection")
    return msgs


def package_versions():
    out = {}
    for m in ("numpy", "torch", "rasterio", "scipy", "pyproj", "matplotlib", "PIL"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = "n/a"
    return out


def validate_scene(name):
    name=clean_name(name)
    if not (UPLOADS/name).is_file(): raise ValueError("SAR scene is missing. Upload it again and select it from the list.")
    info=inspect_sar(name)
    if info["issues"]: raise ValueError("SAR input needs attention: "+" ".join(info["issues"])+" Use a calibrated single-band dB GeoTIFF with a CRS; preprocess raw or linear data outside Proteus.")
    return info


def run_analysis_job(job, p):
    c = core()
    t0 = time.time()
    sar = UPLOADS / clean_name(p["sar"])
    if not sar.is_file():
        raise FileNotFoundError("Select a SAR scene first.")
    validate_scene(sar.name)
    override = str(UPLOADS / clean_name(p["checkpoint_override"])) if p.get("checkpoint_override") else None
    if override and not Path(override).is_file():
        raise ValueError("Selected checkpoint is missing. Select the default model or upload a trusted compatible checkpoint again.")
    if not override and not MODEL_PATH.is_file():
        raise RuntimeError("No trained model is available. Prepare a dataset and train one, or load a custom checkpoint.")
    job["message"] = "Checking references"
    refs = ensure_references(job, [sar.name], p.get("reference_mode", "local"), 0.0, 0.08)
    _, device_note = c.resolve_device(p.get("device", "Auto"))
    job["device"] = device_note
    job_log(job, device_note)
    summary, graphs, zip_path = c.run_proteus_project(
        str(sar), [], override, p.get("mode", "Balanced"), float(p.get("base_threshold", -25)), float(p.get("model_threshold", 0.5)),
        bool(p.get("zero_exclusion", True)), p.get("device", "Auto"), p.get("strategy", "Auto"), int(p.get("tile_size", 2048)),
        progress_cb(job, 0.08, 0.9))
    if isinstance(summary, str) and summary.startswith("ERROR"):
        raise RuntimeError(summary[7:])
    work = Path(zip_path).parent
    results = work / "results"
    report = json.loads((results / "report.json").read_text(encoding="utf-8"))
    rows = add_confidence_index(results / "oil_objects.csv") if (results / "oil_objects.csv").is_file() else []
    bounds = sar_bounds(sar)
    scr = screening(report)
    job["message"] = "Preparing GIS-ready outputs"
    job_log(job, "Preparing GIS-ready outputs (Cloud Optimized GeoTIFF)")
    job["progress"] = 0.92
    import numpy as np
    problems = enrich_geojson(results, rows, work.name, sar.name)
    write_footprint(results, bounds, sar.name, work.name)
    with c.rasterio.open(sar) as s:
        acq = acquisition_time(s)
        step = max(1, int(np.ceil(max(s.width, s.height) / 1500)))
        sample = s.read(1, out_shape=(max(1, s.height // step), max(1, s.width // step)))
        sample = sample[np.isfinite(sample)]
    rng = tuple(np.percentile(sample, [2, 98])) if sample.size else (-30.0, 0.0)
    write_overlay_rgb(sar, results / "suspected_oil_labels.tif", results / "oil_overlay_rgb.tif", rng)
    for n in ("suspected_oil_labels.tif", "model_scores.tif", "oil_object_ids.tif"):
        if (results / n).is_file():
            to_cog(results / n)
    (results / "README_GIS.txt").write_text(GIS_README, encoding="utf-8")
    problems += validate_outputs(results)
    for m in problems:
        job_log(job, "Output check: " + m)
    meta = {"run_id": work.name, "acquisition": acq, "bounds": bounds, "device_requested": p.get("device", "Auto"),
            "device_note": device_note, "screening": scr, "references": refs, "settings": p, "output_checks": problems,
            "run_time_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", "runtime_s": round(time.time() - t0, 1),
            "packages": package_versions()}
    report["proteus_run"] = meta
    (results / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    job["progress"] = 0.96
    job_log(job, "Building PDF report")
    from report_pdf import build_pdf
    build_pdf(results, report, rows, meta)
    shutil.make_archive(str(work / "Proteus_results"), "zip", results)
    top = sorted(rows, key=lambda r: -float(r.get("area_km2") or 0))[:10]
    return {"run_id": work.name, "report": report, "objects": top, "meta": meta,
            "figures": [f"/files/{work.name}/results/{n}" for n in ("oil_preview.png", "backscatter_histogram.png", "object_statistics.png")]}


def _run_file_original(run_id, kind):
    base = OUT / clean_name(run_id)
    if kind == "zip":
        return base / "Proteus_results.zip", "application/zip", f"Proteus_analysis_{base.name}.zip"
    return base / "results" / "analysis_report.pdf", "application/pdf", f"Proteus_report_{base.name}.pdf"
def run_file(run_id, kind):
    path, ctype, suggested = _run_file_original(run_id, kind)
    scene="analysis"
    base=Path(path)
    for cand in [base.parent/"report.json",base.parent/"results"/"report.json",base.parent.parent/"results"/"report.json"]:
        try:
            rep=json.loads(cand.read_text(encoding="utf-8")); name=rep.get("source_filename")
            if isinstance(name,str) and name.strip():
                scene=clean_name(Path(name.replace("\\","/").rsplit("/",1)[-1]).stem); break
        except (OSError,ValueError,TypeError,AttributeError): pass
    if str(path).lower().endswith(".pdf"): suggested=f"ProteusSegNet_Analysis_Report_{scene}.pdf"
    else: suggested=f"ProteusSegNet_Analysis_Results_{scene}.zip"
    return path,ctype,suggested


# ------------------------------------------------------------------ dataset & training
def dataset_scenes(index_path):
    with open(index_path, newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    base = Path(index_path).parent
    for r in rows:
        for k in ("sar_path", "label_path", "validity_path", "report_path"):
            if r.get(k) and not os.path.isabs(r[k]):
                r[k] = str((base / r[k]).resolve())
    return rows


def _dataset_zip_original():
    ds = load_latest_dataset()
    if not ds:
        raise FileNotFoundError("No dataset prepared yet.")
    folder = Path(ds["index_path"]).parent
    archive = shutil.make_archive(str(folder.parent / "Proteus_dataset"), "zip", folder)
    return Path(archive), "application/zip", "Proteus_dataset.zip"
def dataset_zip():
    path,ctype,suggested=_dataset_zip_original()
    return path,ctype,"ProteusSegNet_Dataset.zip"


def prepare_dataset_job(job, p):
    c = core()
    scenes = [clean_name(n) for n in p.get("scenes", [])]
    if not scenes:
        raise ValueError("Tick at least one SAR scene in the list.")
    refs = ensure_references(job, scenes, p.get("reference_mode", "local"), 0.0, 0.1)
    for scene in scenes:
        validate_scene(scene)
    work = OUT / ("dataset_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    sar_dir, ds_dir = work / "sar_inputs", work / "dataset"
    sar_dir.mkdir(parents=True)
    for n in scenes:
        shutil.copy2(UPLOADS / n, sar_dir / n)
    c.WORLD_COVER_FOLDER = str(CACHE)
    job["message"] = "Generating physics pseudo-labels"
    job_log(job, "Generating physics pseudo-labels (CPU)")
    result = c.batch_generate_oil_labels(
        input_dir=str(sar_dir), output_dir=str(ds_dir), device_choice="CPU", mode=p.get("mode", "Balanced"),
        base_threshold_db=float(p.get("base_threshold", -25)), zero_exclusion=bool(p.get("zero_exclusion", True)),
        quick_preview=False, processing_strategy=p.get("strategy", "Auto"), tile_size=int(p.get("tile_size", 2048)),
        progress=progress_cb(job, 0.1, 0.93))
    if result["num_successful"] == 0:
        raise RuntimeError("No dataset scenes were generated. See the log for the failing scene.")
    meta = {"index_path": result["index_path"], "created": dt.datetime.now().isoformat(timespec="seconds"), "settings": p,
            "failures": result.get("failures", [])}
    LATEST_FILE.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    rows = dataset_scenes(result["index_path"])
    job["message"], job["progress"] = "Rendering first preview", 0.95
    import previews
    previews.render_scene(rows[0], CACHE, Path(result["index_path"]).parent / "previews")
    return {"dataset": meta, "scenes": rows, "references": refs}


def train_job(job, p):
    c = core()
    ds = load_latest_dataset()
    if not ds:
        raise RuntimeError("Prepare a dataset first.")
    epochs = int(p.get("epochs", 5))
    _, note = c.resolve_device(p.get("device", "Auto"))
    job["device"] = note
    job_log(job, note)
    job["message"], job["progress"] = "Initialising training", 0.05
    path, history = c.train_proteus_model(ds["index_path"], epochs=epochs, device_choice=p.get("device", "Auto"))
    import previews
    tdir = OUT / "training"
    tdir.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    loss = [float(x) for x in history]
    fig = previews.loss_figure(loss, tdir / f"training_loss_{stamp}.png")
    return {"checkpoint": str(path), "loss": loss, "figure": f"/files/training/{fig.name}", "device_note": c.DEVICE_NOTE,
            "epochs": epochs, "final_loss": loss[-1] if loss else None, "first_loss": loss[0] if loss else None,
            "finished": dt.datetime.now().isoformat(timespec="minutes")}
