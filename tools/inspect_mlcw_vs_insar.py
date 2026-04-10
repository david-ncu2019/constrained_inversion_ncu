"""
inspect_mlcw_vs_insar.py

For each MLCW station: plot cumulative InSAR surface displacement (top) and
cumulative MLCW ring-by-ring compaction with per-depth colour coding (bottom).

Run from: D:\110_PROJECT_002\constrained_inversion_scripts\
    python inspect_mlcw_vs_insar.py
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.loader import build_real_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "my_input_data"
OUT_DIR = SCRIPT_DIR / "inspection_figs"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
dataset, config, meta = build_real_dataset(DATA_DIR)

n_stations: int = config.n_pixels   # 35
n_epochs: int   = config.n_epochs   # 83
n_layers: int   = config.n_layers   # 30

station_names: list = meta["station_names"]          # len 35
depths_m: list      = meta["valid_depths_m"]         # len 30, values 10..300

# ---------------------------------------------------------------------------
# Reshape raw flat arrays
# d_insar: epoch-major -> (n_epochs, n_stations) -> (n_stations, n_epochs)
# w:       epoch-major -> (n_epochs, n_stations * n_layers) -> (n_stations, n_layers, n_epochs)
# ---------------------------------------------------------------------------
d_insar_2d = dataset.d_insar.reshape(n_epochs, n_stations).T          # (n_stations, n_epochs)
w_3d = dataset.w.reshape(n_epochs, n_stations * n_layers)             # (n_epochs, n_stations*n_layers)
# rearrange to (n_stations, n_layers, n_epochs)
w_3d = w_3d.reshape(n_epochs, n_stations, n_layers)                   # (n_epochs, n_stations, n_layers)
w_3d = w_3d.transpose(1, 2, 0)                                        # (n_stations, n_layers, n_epochs)

# ---------------------------------------------------------------------------
# Colour mapping for depth layers
# shallow (10 m) = light end, deep (300 m) = dark end of plasma
# ---------------------------------------------------------------------------
cmap_name = "plasma"
cmap = cm.get_cmap(cmap_name)
depth_colours = [cmap(v) for v in np.linspace(0, 1, n_layers)]

norm = Normalize(vmin=float(depths_m[0]), vmax=float(depths_m[-1]))
scalar_mappable = ScalarMappable(norm=norm, cmap=cmap_name)
scalar_mappable.set_array([])

month_axis = np.arange(1, n_epochs + 1)  # 1, 2, ..., 83

# ---------------------------------------------------------------------------
# Per-station figures
# ---------------------------------------------------------------------------
for s_idx, station in enumerate(station_names):

    insar_inc   = d_insar_2d[s_idx]                   # (n_epochs,)
    insar_cum   = np.cumsum(insar_inc)                 # cumulative

    well_inc    = w_3d[s_idx]                          # (n_layers, n_epochs)
    well_cum    = np.cumsum(well_inc, axis=1)          # (n_layers, n_epochs)
    sum_cum     = np.cumsum(well_inc.sum(axis=0))      # sum across layers, then cumulative

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(16, 9),
        constrained_layout=True,
    )

    # --- Top: InSAR cumulative ---
    ax_top.plot(month_axis, insar_cum, color="steelblue", linewidth=2)
    ax_top.set_title(f"{station} — InSAR surface displacement", fontsize=16)
    ax_top.set_xlabel("Months", fontsize=14)
    ax_top.set_ylabel("Cumulative displacement (mm)", fontsize=14)
    ax_top.tick_params(labelsize=12)
    ax_top.grid(True, linestyle="--", alpha=0.4)

    # --- Bottom: MLCW per-depth cumulative + sum ---
    for l_idx in range(n_layers):
        ax_bot.plot(
            month_axis,
            well_cum[l_idx],
            color=depth_colours[l_idx],
            linewidth=1,
            alpha=0.85,
        )

    ax_bot.plot(
        month_axis,
        sum_cum,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="Sum(all layers)",
        zorder=5,
    )

    ax_bot.set_title(
        f"{station} — MLCW ring-by-ring compaction (0–300 m)", fontsize=16
    )
    ax_bot.set_xlabel("Months", fontsize=14)
    ax_bot.set_ylabel("Cumulative compaction (mm)", fontsize=14)
    ax_bot.tick_params(labelsize=12)
    ax_bot.grid(True, linestyle="--", alpha=0.4)
    ax_bot.legend(fontsize=12, loc="upper left")

    cbar = fig.colorbar(scalar_mappable, ax=ax_bot, pad=0.02, aspect=30)
    cbar.set_label("Depth (m)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # --- Save ---
    out_path = OUT_DIR / f"{station}_mlcw_vs_insar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: inspection_figs/{station}_mlcw_vs_insar.png")

print(f"\nDone. {n_stations} figures written to {OUT_DIR}")
