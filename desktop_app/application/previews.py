"""Dataset previews, WorldCover / labelling graphs and the training-loss figure."""
import json, math
from pathlib import Path
import numpy as np
import rasterio
from affine import Affine
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

NAVY, PANEL, AMBER, TEXT, MUTED = "#0b1626", "#10223a", "#e8890c", "#e6eef8", "#8aa0b8"
WATER, LAND, UNK = "#2a7fb8", "#8a7a56", "#33465c"
KNOWN = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
WC_COLORS = ListedColormap([UNK, LAND, WATER])
LABEL_COLORS = ListedColormap([WATER, AMBER, UNK])


def _style(ax, title):
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=8)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#243a57")


def _save(fig, path):
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def _image(arr, cmap, title, path, vmin=None, vmax=None, legend=None):
    fig, ax = plt.subplots(figsize=(5.4, 4.8), facecolor=NAVY)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    _style(ax, title)
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(handles=[Patch(color=c, label=l) for l, c in legend], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(legend),
                  fontsize=7.5, frameon=False, labelcolor=TEXT)
    return _save(fig, path)


def render_scene(row, cache_dir, out_dir, max_dim=900):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sid = row["image_id"]
    with rasterio.open(row["sar_path"]) as s, rasterio.open(row["label_path"]) as lab:
        k = max(max(s.width, s.height) / max_dim, 1.0)
        ow, oh = max(1, int(s.width / k)), max(1, int(s.height / k))
        sar = s.read(1, out_shape=(oh, ow), resampling=Resampling.nearest).astype("float32")
        labels = lab.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
        tr = s.transform * Affine.scale(s.width / ow, s.height / oh)
        west, south, east, north = transform_bounds(s.crs, "EPSG:4326", *s.bounds, densify_pts=21)
        wc = np.zeros((oh, ow), dtype=np.uint8)
        for la in range(int(math.floor(south / 3) * 3), int(math.floor((north - 1e-9) / 3) * 3) + 1, 3):
            for lo in range(int(math.floor(west / 3) * 3), int(math.floor((east - 1e-9) / 3) * 3) + 1, 3):
                nm = f"ESA_WorldCover_10m_2021_v200_{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}_Map.tif"
                fp = Path(cache_dir) / nm
                if not fp.is_file():
                    continue
                with rasterio.open(fp) as w, WarpedVRT(w, crs=s.crs, transform=tr, width=ow, height=oh, src_nodata=0, nodata=0,
                                                       resampling=Resampling.nearest) as v:
                    vals = v.read(1)
                    wc[np.isin(vals, KNOWN) & (vals != 80)] = 1
                    wc[vals == 80] = 2
        counts = np.zeros(3, dtype=np.int64)
        for _, win in lab.block_windows(1):
            blk = lab.read(1, window=win)
            counts += [(blk == 0).sum(), (blk == 1).sum(), (blk == 255).sum()]
    finite = sar[np.isfinite(sar)]
    lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (-30, 0))
    gray = plt.get_cmap("gray")
    _image(sar, gray, "Calibrated SAR (dB, sampled)", out_dir / f"{sid}_sar.png", lo, hi)
    _image(wc, WC_COLORS, "Aligned WorldCover reference", out_dir / f"{sid}_worldcover.png", 0, 2,
           [("Unknown", UNK), ("Non-water", LAND), ("Water", WATER)])
    _image(np.where(labels == 255, 2, labels), LABEL_COLORS, "Pseudo-label", out_dir / f"{sid}_labels.png", 0, 2,
           [("Evaluated water", WATER), ("Suspected oil", AMBER), ("Excluded", UNK)])
    fig, ax = plt.subplots(figsize=(5.4, 4.8), facecolor=NAVY)
    ax.imshow(sar, cmap=gray, vmin=lo, vmax=hi, interpolation="nearest")
    ax.imshow(np.ma.masked_where(labels != 1, np.ones_like(labels)), cmap=ListedColormap([AMBER]), alpha=0.9, interpolation="nearest")
    ax.imshow(np.ma.masked_where(labels != 255, np.ones_like(labels)), cmap=ListedColormap([NAVY]), alpha=0.45, interpolation="nearest")
    _style(ax, "Overlay: suspected oil on SAR"); ax.set_xticks([]); ax.set_yticks([])
    _save(fig, out_dir / f"{sid}_overlay.png")

    tot_wc = max(1, wc.size)
    share = [(wc == 2).sum() / tot_wc, (wc == 1).sum() / tot_wc, (wc == 0).sum() / tot_wc]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=NAVY)
    bars = axs[0].barh(["Water", "Non-water", "Unknown"], [100 * x for x in share], color=[WATER, LAND, UNK])
    axs[0].invert_yaxis(); axs[0].set_xlim(0, 118)
    axs[0].set_xlabel("Share of sampled display grid (%)", color=MUTED, fontsize=9)
    for r_, x in zip(bars, share):
        axs[0].text(r_.get_width() + 1.5, r_.get_y() + r_.get_height() / 2, f"{100 * x:.1f}%", va="center", color=TEXT, fontsize=8)
    _style(axs[0], "WorldCover class composition")
    tot = max(1, int(counts.sum()))
    axs[1].pie([int(c) for c in counts], colors=[WATER, AMBER, UNK], startangle=90, wedgeprops={"width": 0.38, "edgecolor": NAVY})
    axs[1].set_title("Reference coverage (full resolution)", color=TEXT, fontsize=11, loc="left")
    axs[1].legend([f"Evaluated water {100 * counts[0] / tot:.2f}%", f"Suspected oil {100 * counts[1] / tot:.3f}%", f"Excluded {100 * counts[2] / tot:.2f}%"],
                  loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, labelcolor=TEXT, fontsize=8)
    _save(fig, out_dir / f"{sid}_worldcover_chart.png")

    wv, ov = sar[(labels == 0) & np.isfinite(sar)], sar[(labels == 1) & np.isfinite(sar)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=NAVY)
    b = axs[0].bar(["Evaluated water", "Suspected oil", "Excluded"], counts, color=[WATER, AMBER, UNK])
    axs[0].set_yscale("symlog"); axs[0].set_ylabel("Pixels (log scale)", color=MUTED, fontsize=9)
    for r_, v in zip(b, counts):
        axs[0].text(r_.get_x() + r_.get_width() / 2, max(v, 1) * 1.2, f"{int(v):,}", ha="center", color=TEXT, fontsize=7.5)
    _style(axs[0], "Pseudo-label composition")
    if finite.size:
        bins = np.linspace(lo - 5, min(hi + 5, finite.max()), 50)
        axs[1].hist(wv, bins=bins, color=WATER, alpha=0.8, density=True, label="Evaluated water")
        if ov.size:
            axs[1].hist(ov, bins=bins, color=AMBER, alpha=0.85, density=True, label="Suspected oil")
        axs[1].legend(frameon=False, labelcolor=TEXT, fontsize=8)
    axs[1].set_xlabel("Backscatter (dB, sampled)", color=MUTED, fontsize=9); axs[1].set_ylabel("Density", color=MUTED, fontsize=9)
    _style(axs[1], "Backscatter by label class")
    _save(fig, out_dir / f"{sid}_label_chart.png")
    stats = {"evaluated_water_px": int(counts[0]), "oil_px": int(counts[1]), "excluded_px": int(counts[2]),
             "wc_water_pct": round(100 * share[0], 2), "wc_nonwater_pct": round(100 * share[1], 2), "wc_unknown_pct": round(100 * share[2], 2)}
    (out_dir / f"{sid}_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return {"image_id": sid, "stats": stats}


def loss_figure(history, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor="white")
    ax.plot(range(1, len(history) + 1), history, color="#0f766e", linewidth=1.8, marker="o", markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Training loss by epoch", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return Path(path)
