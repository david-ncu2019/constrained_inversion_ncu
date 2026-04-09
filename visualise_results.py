"""
visualise_results.py — Diagnostic and publication-quality plots for the CRFP pipeline.

Reads completed pipeline outputs and produces 12 plot types covering inversion quality,
spatial CV, temporal CV, final grid maps, and depth-time profiles.

Usage
-----
  uv run python visualise_results.py [options]

Options
-------
  --output-dir PATH       Pipeline output directory (default: output/)
  --vis-dir PATH          Where to save plots (default: output/visualisations/)
  --map-month INT         Month index (0-based) for map snapshots (default: last available)
  --profile-stations STR  Comma-separated station names for depth-time heatmaps.
                          Default: 3 stations with largest deep-residual fraction.
  --dpi INT               Figure DPI (default: 120)
  --no-maps               Skip Stage-4 map plots (fast; skips loading the large NetCDF)

Outputs (under --vis-dir)
--------------------------
  inversion/
    deep_residual_scatter.png
    deep_residual_timeseries.png
  cv_spatial/
    station_error_map.png
    loso_per_layer_bar.png
    depth_profile_worst_station.png
  cv_temporal/
    rmse_heatmap_fold_x_layer.png
    timeseries_fold3_<station>.png   (one per worst-5 station)
  maps/
    compaction_depth_intervals_month<N>.png
    deep_residual_map_month<N>.png
    uncertainty_map_50m_month<N>.png
    low_confidence_mask.png
  profiles/
    depth_time_heatmap_<station>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_paths(args: argparse.Namespace) -> dict:
    out = Path(args.output_dir)
    vis = Path(args.vis_dir)
    return {
        "output_dir": out,
        "vis_dir": vis,
        "inversion_npz": out / "real_m_est.npz",
        "gp_nc": out / "depth_weights_gp_grid.nc",
        "spatial_csv": out / "cv_spatial_gp" / "loso_per_layer.csv",
        "spatial_station_csv": out / "cv_spatial_gp" / "loso_per_station.csv",
        "spatial_npz": out / "cv_spatial_gp" / "loso_gp_predictions.npz",
        "temporal_csv": out / "cv_temporal" / "temporal_cv_per_layer.csv",
        "temporal_station_csv": out / "cv_temporal" / "temporal_cv_per_station_layer.csv",
        "temporal_arrays_npz": out / "cv_temporal" / "temporal_cv_arrays.npz",
        "inversion_dir": vis / "inversion",
        "cv_spatial_dir": vis / "cv_spatial",
        "cv_temporal_dir": vis / "cv_temporal",
        "maps_dir": vis / "maps",
        "profiles_dir": vis / "profiles",
    }


def _makedirs(paths: dict) -> None:
    for key in ("inversion_dir", "cv_spatial_dir", "cv_temporal_dir", "maps_dir", "profiles_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_inversion_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_netcdf(path: Path):
    """Return xarray Dataset or None if file missing."""
    if not path.exists():
        print(f"  [SKIP] NetCDF not found: {path}")
        return None
    import xarray as xr
    return xr.open_dataset(path)


# ---------------------------------------------------------------------------
# 1–2  Inversion diagnostics
# ---------------------------------------------------------------------------


def plot_inversion_diagnostics(npz: dict, paths: dict, dpi: int) -> None:
    station_names = npz["station_names"].astype(str)
    insar_coverage = np.array(npz["insar_coverage"], dtype=float)
    depth_weights = np.array(npz["depth_weights"], dtype=float)   # (n_st, n_lay)
    month_labels = npz["month_labels"].astype(str)
    n_st = len(station_names)

    # ── 1. Deep-residual scatter ──────────────────────────────────────────
    sort_idx = np.argsort(insar_coverage)
    sorted_cov = insar_coverage[sort_idx]
    sorted_names = station_names[sort_idx]
    flagged = sorted_cov < 0.85

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = ["#d73027" if f else "#4575b4" for f in flagged]
    ax.scatter(np.arange(n_st), sorted_cov, c=colors, s=40, zorder=3)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="100 %")
    ax.axhline(0.9, color="orange", lw=0.8, ls="--", label="90 %")
    ax.axhline(0.85, color="red", lw=0.8, ls="--", label="85 % threshold")
    ax.set_xlabel("Station (sorted by coverage)")
    ax.set_ylabel("InSAR coverage  (0–300 m / total)")
    ax.set_title("Shallow coverage fraction per station")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, n_st - 0.5)
    ax.set_ylim(0, 1.1)

    ax2 = axes[1]
    deep_frac = 1.0 - insar_coverage[sort_idx]
    bar_colors = ["#d73027" if f else "#91bfdb" for f in flagged]
    ax2.bar(np.arange(n_st), deep_frac, color=bar_colors)
    ax2.set_xlabel("Station (sorted by coverage)")
    ax2.set_ylabel("Deep-residual fraction  (>300 m)")
    ax2.set_title("Deep compaction fraction per station")
    ax2.set_xlim(-0.5, n_st - 0.5)
    # Annotate top-3 worst
    worst3 = np.argsort(deep_frac)[-3:]
    for idx in worst3:
        ax2.text(idx, deep_frac[idx] + 0.005, sorted_names[idx], ha="center",
                 fontsize=6, rotation=60)

    plt.tight_layout()
    out = paths["inversion_dir"] / "deep_residual_scatter.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 2. Deep-residual time series ──────────────────────────────────────
    # deep_residual_fraction[s] = 1 - sum_l(depth_weights[s,l])
    weight_sum = depth_weights.sum(axis=1)        # (n_st,)
    deep_frac_all = 1.0 - weight_sum              # (n_st,)

    # Pick 4 representative stations: highest/lowest coverage + 2 intermediate
    rep_idx = [
        int(np.argmax(weight_sum)),   # most shallow
        int(np.argmin(weight_sum)),   # most deep
        int(np.argsort(weight_sum)[n_st // 3]),
        int(np.argsort(weight_sum)[2 * n_st // 3]),
    ]
    rep_idx = list(dict.fromkeys(rep_idx))  # deduplicate, preserve order

    n_ep = len(month_labels)
    t = np.arange(n_ep)

    fig, ax = plt.subplots(figsize=(10, 4))
    ls_cycle = ["-", "--", "-.", ":"]
    for k, s_idx in enumerate(rep_idx):
        # cumulative deep residual = deep_frac * cumulative InSAR
        # We don't have insar_cum per station here, so plot weight-sum deficit as
        # a constant fraction bar; use m_est to derive cumulative station signal.
        m_est = np.array(npz["m_est"], dtype=float)  # (n_ep, n_st, n_lay)
        cum_shallow = np.cumsum(m_est[:, s_idx, :].sum(axis=1))  # (n_ep,)
        # deep residual (estimated) = shallow_cum / weight_sum[s_idx] * deep_frac[s_idx]
        if weight_sum[s_idx] > 0:
            cum_insar_est = cum_shallow / weight_sum[s_idx]
        else:
            cum_insar_est = cum_shallow * 0
        cum_deep_est = cum_insar_est * deep_frac_all[s_idx]
        label = f"{station_names[s_idx]}  (deep frac={deep_frac_all[s_idx]:.2f})"
        ax.plot(t, cum_deep_est, ls=ls_cycle[k % 4], label=label, lw=1.5)

    ax.set_xlabel("Month index")
    ax.set_ylabel("Estimated cumulative deep residual (mm)")
    ax.set_title("Cumulative deep-residual signal at representative stations")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    out = paths["inversion_dir"] / "deep_residual_timeseries.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 3–5  Spatial CV
# ---------------------------------------------------------------------------


def plot_spatial_cv(paths: dict, npz: dict, dpi: int) -> None:
    spatial_csv = paths["spatial_csv"]
    spatial_station_csv = paths["spatial_station_csv"]
    spatial_npz_path = paths["spatial_npz"]

    if not spatial_csv.exists():
        print("  [SKIP] Spatial CV CSVs not found.")
        return

    df_layer = pd.read_csv(spatial_csv)

    # ── 3. Station error map ──────────────────────────────────────────────
    if spatial_station_csv.exists():
        df_st = pd.read_csv(spatial_station_csv)
        x_km = np.array(npz["x_twd97"], dtype=float) / 1000.0
        y_km = np.array(npz["y_twd97"], dtype=float) / 1000.0
        st_names = npz["station_names"].astype(str)

        # Merge on station_name
        st_error = {}
        for _, row in df_st.iterrows():
            st_error[row["station_name"]] = (row["total_col_error"], row["total_col_abs_error"])

        errors = np.array([st_error.get(n, (0.0, 0.0))[0] for n in st_names])
        abs_errors = np.array([st_error.get(n, (0.0, 0.0))[1] for n in st_names])

        vmax = np.nanpercentile(np.abs(errors), 95)
        vmax = max(vmax, 1e-6)

        fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter(
            x_km, y_km,
            c=errors,
            s=60 + 400 * abs_errors / (abs_errors.max() + 1e-9),
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            edgecolors="k", linewidths=0.4,
            zorder=3,
        )
        plt.colorbar(sc, ax=ax, label="LOO error (pred − true, weight units)")
        ax.set_xlabel("Easting (km, TWD97)")
        ax.set_ylabel("Northing (km, TWD97)")
        ax.set_title("Spatial CV — per-station leave-one-out error\n(point size ∝ |error|)")

        worst3_idx = np.argsort(abs_errors)[-3:]
        for idx in worst3_idx:
            ax.annotate(
                st_names[idx],
                (x_km[idx], y_km[idx]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points",
            )

        plt.tight_layout()
        out = paths["cv_spatial_dir"] / "station_error_map.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  Saved: {out}")

    # ── 4. Per-layer RMSE bar chart ───────────────────────────────────────
    depths = df_layer["depth_m"].values
    rmse = df_layer["rmse"].values
    mae = df_layer["mae"].values if "mae" in df_layer.columns else None

    fig, ax = plt.subplots(figsize=(6, 8))
    y = np.arange(len(depths))
    ax.barh(y, rmse, color="#2166ac", alpha=0.85, label="RMSE")
    if mae is not None:
        ax.barh(y, mae, color="#92c5de", alpha=0.7, label="MAE")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d} m" for d in depths], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Error (dimensionless weight units)")
    ax.set_title("Spatial CV — per-layer RMSE and MAE")
    ax.legend()
    plt.tight_layout()
    out = paths["cv_spatial_dir"] / "loso_per_layer_bar.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 5. Depth profile for worst station ───────────────────────────────
    if spatial_npz_path.exists() and spatial_station_csv.exists():
        pred_data = np.load(spatial_npz_path, allow_pickle=True)
        mean_pred = pred_data["mean_pred"]     # (n_st, n_lay)
        std_pred = pred_data["std_pred"]       # (n_st, n_lay)
        true_w = pred_data["true_weights"]     # (n_st, n_lay)
        pred_st_names = pred_data["station_names"].astype(str)
        pred_depths = pred_data["valid_depths_m"]

        df_st2 = pd.read_csv(spatial_station_csv)
        worst_name = df_st2.loc[df_st2["total_col_abs_error"].idxmax(), "station_name"]
        w_idx = np.where(pred_st_names == worst_name)[0]
        if len(w_idx) == 0:
            print(f"  [WARN] Station '{worst_name}' not found in loso_gp_predictions.npz")
            return
        w_idx = int(w_idx[0])

        fig, ax = plt.subplots(figsize=(5, 8))
        ax.plot(true_w[w_idx], pred_depths, "b-", lw=1.5, label="True (full-data inversion)")
        ax.plot(mean_pred[w_idx], pred_depths, "r--", lw=1.5, label="GP prediction (LOO mean)")
        ax.fill_betweenx(
            pred_depths,
            mean_pred[w_idx] - std_pred[w_idx],
            mean_pred[w_idx] + std_pred[w_idx],
            color="red", alpha=0.2, label="±1σ",
        )
        ax.invert_yaxis()
        ax.set_xlabel("Depth weight (dimensionless)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Spatial CV — depth profile: {worst_name}\n(worst LOO station)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = paths["cv_spatial_dir"] / "depth_profile_worst_station.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 6–7  Temporal CV
# ---------------------------------------------------------------------------


def plot_temporal_cv(paths: dict, dpi: int) -> None:
    temporal_csv = paths["temporal_csv"]
    temporal_station_csv = paths["temporal_station_csv"]
    temporal_arrays_npz = paths["temporal_arrays_npz"]

    if not temporal_csv.exists():
        print("  [SKIP] Temporal CV CSVs not found.")
        return

    df = pd.read_csv(temporal_csv)

    # ── 6. RMSE heatmap (fold × depth) ───────────────────────────────────
    folds = sorted(df["fold"].unique())
    depths = sorted(df["depth_m"].unique())
    matrix = np.full((len(depths), len(folds)), np.nan)
    for fi, fold in enumerate(folds):
        for di, depth in enumerate(depths):
            rows = df[(df["fold"] == fold) & (df["depth_m"] == depth)]
            if not rows.empty:
                matrix[di, fi] = rows["rmse"].values[0]

    fig, ax = plt.subplots(figsize=(5, 9))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0,
                   vmax=np.nanpercentile(matrix, 95))
    plt.colorbar(im, ax=ax, label="RMSE (mm)")
    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([f"{d} m" for d in depths], fontsize=7)
    ax.set_title("Temporal CV RMSE (mm)\n(depth × fold)")

    # Annotate cells
    for di in range(len(depths)):
        for fi in range(len(folds)):
            val = matrix[di, fi]
            if not np.isnan(val):
                ax.text(fi, di, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color="k" if val < np.nanmax(matrix) * 0.6 else "w")

    plt.tight_layout()
    out = paths["cv_temporal_dir"] / "rmse_heatmap_fold_x_layer.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 7. Fold-3 time series for 5 worst stations ────────────────────────
    if not temporal_station_csv.exists() or not temporal_arrays_npz.exists():
        print("  [SKIP] temporal_cv_per_station_layer.csv or temporal_cv_arrays.npz not found.")
        return

    df_st = pd.read_csv(temporal_station_csv)
    fold3_st = df_st[df_st["fold"] == 3]
    worst_stations = (
        fold3_st.groupby("station")["rmse"].mean()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    arr = np.load(temporal_arrays_npz, allow_pickle=True)
    pred_all = arr["pred"]    # (n_folds, max_val_len, n_st, n_lay)
    true_all = arr["true"]    # same shape
    st_names = arr["station_names"].astype(str)
    depths_arr = arr["valid_depths_m"]

    # Fold 3 is index 2
    pred_f3 = pred_all[2]   # (max_val_len, n_st, n_lay)
    true_f3 = true_all[2]

    target_depths = [50, 100, 150, 200, 250, 300]
    for sname in worst_stations:
        s_idx_arr = np.where(st_names == sname)[0]
        if len(s_idx_arr) == 0:
            continue
        s_idx = int(s_idx_arr[0])

        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        axes = axes.flatten()

        for panel_i, td in enumerate(target_depths):
            d_idx_arr = np.where(depths_arr == td)[0]
            ax = axes[panel_i]
            if len(d_idx_arr) == 0:
                ax.set_visible(False)
                continue
            d_idx = int(d_idx_arr[0])
            n_t = pred_f3.shape[0]
            t = np.arange(n_t)
            p = pred_f3[:, s_idx, d_idx]
            o = true_f3[:, s_idx, d_idx]
            valid = ~np.isnan(o)
            ax.plot(t[valid], o[valid], "b-", lw=1.5, label="Observed")
            ax.plot(t[valid], p[valid], "r--", lw=1.5, label="Predicted")
            ax.set_title(f"{td} m", fontsize=9)
            ax.set_ylabel("Cum. compaction (mm)", fontsize=7)
            if panel_i == 0:
                ax.legend(fontsize=7)

        fig.suptitle(f"Temporal CV — Fold 3 — {sname}", fontsize=11)
        plt.tight_layout()
        safe_name = sname.replace("/", "_").replace(" ", "_")
        out = paths["cv_temporal_dir"] / f"timeseries_fold3_{safe_name}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 8–11  Grid maps
# ---------------------------------------------------------------------------


def _map_extent(X: np.ndarray, Y: np.ndarray) -> list:
    """Return imshow extent in km."""
    return [X.min() / 1000, X.max() / 1000, Y.min() / 1000, Y.max() / 1000]


def plot_maps(ds, npz: dict, map_month: int | None, paths: dict, dpi: int) -> None:
    X = ds["X"].values
    Y = ds["Y"].values
    layers = ds["layer"].values          # depths in metres
    n_time = ds.dims["time"]

    month_idx = map_month if map_month is not None else n_time - 1
    month_idx = min(month_idx, n_time - 1)
    month_label = str(ds["time"].values[month_idx])

    x_km = np.array(npz["x_twd97"], dtype=float) / 1000.0
    y_km = np.array(npz["y_twd97"], dtype=float) / 1000.0
    extent = _map_extent(X, Y)

    # ── 8. Layer compaction maps (3 depth intervals) ──────────────────────
    ranges = [(0, 50), (50, 150), (150, 300)]
    range_labels = ["0–50 m", "50–150 m", "150–300 m"]

    comp = ds["layer_compaction"].values   # (time, layer, Y, X)
    comp_t = comp[month_idx]               # (layer, Y, X)

    panels = []
    for lo, hi in ranges:
        mask = (layers >= lo) & (layers <= hi)
        if not mask.any():
            panels.append(np.zeros((comp_t.shape[1], comp_t.shape[2])))
        else:
            panels.append(np.nansum(comp_t[mask], axis=0))

    vmax = max(np.nanpercentile(np.abs(p), 98) for p in panels)
    vmax = max(vmax, 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, panel, label in zip(axes, panels, range_labels):
        im = ax.imshow(
            panel, origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal",
        )
        ax.scatter(x_km, y_km, s=15, c="k", edgecolors="w", linewidths=0.5, zorder=3)
        plt.colorbar(im, ax=ax, label="mm", shrink=0.7)
        ax.set_title(f"Compaction {label}")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")

    fig.suptitle(f"Layer-wise cumulative compaction — {month_label}", fontsize=12)
    plt.tight_layout()
    out = paths["maps_dir"] / f"compaction_depth_intervals_month{month_idx:03d}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 9. Deep residual map ──────────────────────────────────────────────
    dr = ds["deep_residual"].values[month_idx]   # (Y, X)
    n_neg = int(np.sum(dr < -1.0))
    vr = np.nanpercentile(np.abs(dr), 98)
    vr = max(vr, 0.1)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        dr, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-vr, vmax=vr, aspect="equal",
    )
    ax.scatter(x_km, y_km, s=15, c="k", edgecolors="w", linewidths=0.5, zorder=3)
    plt.colorbar(im, ax=ax, label="mm")
    title = f"Deep residual (>300 m) — {month_label}"
    if n_neg > 0:
        title += f"\n⚠ {n_neg} pixels with residual < −1 mm (unphysical)"
    ax.set_title(title)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    plt.tight_layout()
    out = paths["maps_dir"] / f"deep_residual_map_month{month_idx:03d}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 10. Uncertainty map ───────────────────────────────────────────────
    # Use depth = 50 m (first layer at or above 50 m)
    d50_arr = np.where(layers >= 50)[0]
    d50 = int(d50_arr[0]) if len(d50_arr) > 0 else 0
    depth_label = f"{int(layers[d50])} m"

    has_total = "compaction_total_std" in ds
    if has_total:
        std_gp = ds["compaction_std"].values[month_idx, d50]
        std_tot = ds["compaction_total_std"].values[month_idx, d50]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, data, label in zip(
            axes,
            [std_gp, std_tot],
            ["GP-only std (σ_gp)", "Combined std (σ_total)"],
        ):
            vmax_u = np.nanpercentile(data, 98)
            vmax_u = max(vmax_u, 0.01)
            im = ax.imshow(
                data, origin="lower", extent=extent,
                cmap="Oranges", vmin=0, vmax=vmax_u, aspect="equal",
            )
            ax.scatter(x_km, y_km, s=15, c="k", edgecolors="w", linewidths=0.5, zorder=3)
            plt.colorbar(im, ax=ax, label="mm")
            ax.set_title(label)
            ax.set_xlabel("Easting (km)")
            ax.set_ylabel("Northing (km)")
    else:
        std_gp = ds["compaction_std"].values[month_idx, d50]
        fig, ax = plt.subplots(figsize=(7, 7))
        vmax_u = np.nanpercentile(std_gp, 98)
        vmax_u = max(vmax_u, 0.01)
        im = ax.imshow(
            std_gp, origin="lower", extent=extent,
            cmap="Oranges", vmin=0, vmax=vmax_u, aspect="equal",
        )
        ax.scatter(x_km, y_km, s=15, c="k", edgecolors="w", linewidths=0.5, zorder=3)
        plt.colorbar(im, ax=ax, label="mm")
        ax.set_title("GP-only std (σ_gp) — CV not available")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")

    fig.suptitle(f"Uncertainty at {depth_label} — {month_label}", fontsize=11)
    plt.tight_layout()
    out = paths["maps_dir"] / f"uncertainty_map_{depth_label.replace(' ', '')}_{month_idx:03d}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 11. Low-confidence mask ───────────────────────────────────────────
    lc = ds["low_confidence"].values   # (Y, X)
    n_lc = int(lc.sum())
    total_px = lc.size
    pct = 100.0 * n_lc / total_px

    fig, ax = plt.subplots(figsize=(7, 7))
    # White = ok, grey = flagged
    cmap_lc = mcolors.ListedColormap(["white", "#aaaaaa"])
    ax.imshow(lc, origin="lower", extent=extent, cmap=cmap_lc,
              vmin=0, vmax=1, aspect="equal")
    ax.scatter(x_km, y_km, s=20, c="#e31a1c", edgecolors="k", linewidths=0.4, zorder=3,
               label="MLCW stations")
    ax.set_title(
        f"Low-confidence mask\n"
        f"{n_lc} / {total_px} pixels flagged ({pct:.1f} %) "
        f"— grey = GP std exceeds threshold"
    )
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = paths["maps_dir"] / "low_confidence_mask.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 12  Depth-time heatmaps
# ---------------------------------------------------------------------------


def plot_profiles(
    ds,
    npz: dict,
    profile_station_names: list[str],
    paths: dict,
    dpi: int,
) -> None:
    if not profile_station_names:
        return

    X = ds["X"].values
    Y = ds["Y"].values
    layers = ds["layer"].values
    comp = ds["layer_compaction"].values   # (time, layer, Y, X)
    n_time = comp.shape[0]

    st_names = npz["station_names"].astype(str)
    x_twd97 = np.array(npz["x_twd97"], dtype=float)
    y_twd97 = np.array(npz["y_twd97"], dtype=float)

    for sname in profile_station_names:
        s_idx_arr = np.where(st_names == sname)[0]
        if len(s_idx_arr) == 0:
            print(f"  [WARN] Station '{sname}' not found in real_m_est.npz — skipping profile.")
            continue
        s_idx = int(s_idx_arr[0])

        sx = x_twd97[s_idx]
        sy = y_twd97[s_idx]

        # Nearest grid pixel
        col = int(np.argmin(np.abs(X - sx)))
        row = int(np.argmin(np.abs(Y - sy)))

        profile = comp[:, :, row, col]   # (n_time, n_layers)

        vmax = np.nanpercentile(np.abs(profile), 98)
        vmax = max(vmax, 0.1)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            profile.T, origin="upper", aspect="auto",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            extent=[0, n_time, layers[-1], layers[0]],
        )
        plt.colorbar(im, ax=ax, label="Cumulative compaction (mm)")
        ax.set_xlabel("Month index")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Depth–time compaction profile: {sname}")
        plt.tight_layout()
        safe_name = sname.replace("/", "_").replace(" ", "_")
        out = paths["profiles_dir"] / f"depth_time_heatmap_{safe_name}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate diagnostic and publication plots for the CRFP inversion pipeline."
    )
    p.add_argument("--output-dir", default="output/", type=Path,
                   help="Pipeline output directory (default: output/).")
    p.add_argument("--vis-dir", default=None, type=Path,
                   help="Plot output directory (default: <output-dir>/visualisations/).")
    p.add_argument("--map-month", default=None, type=int,
                   help="0-based month index for map snapshots (default: last available).")
    p.add_argument("--profile-stations", default="", type=str,
                   help="Comma-separated station names for depth-time heatmaps.")
    p.add_argument("--dpi", default=120, type=int,
                   help="Figure DPI (default: 120).")
    p.add_argument("--no-maps", action="store_true", default=False,
                   help="Skip map plots (skips loading the large NetCDF).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    vis_dir = Path(args.vis_dir) if args.vis_dir is not None else output_dir / "visualisations"
    args.vis_dir = vis_dir  # normalise

    paths = resolve_paths(args)
    _makedirs(paths)

    # ── Load core inversion results ───────────────────────────────────────
    if not paths["inversion_npz"].exists():
        print(f"ERROR: Inversion NPZ not found: {paths['inversion_npz']}")
        print("Run the pipeline (Stage 2) before visualising.")
        raise SystemExit(1)

    print("Loading inversion NPZ...")
    npz = load_inversion_npz(paths["inversion_npz"])

    # ── Determine profile stations ────────────────────────────────────────
    if args.profile_stations.strip():
        profile_stations = [s.strip() for s in args.profile_stations.split(",")]
    else:
        # Default: 3 stations with largest deep-residual fraction
        weight_sum = np.array(npz["depth_weights"], dtype=float).sum(axis=1)
        deep_frac = 1.0 - weight_sum
        worst3_idx = np.argsort(deep_frac)[-3:]
        profile_stations = list(npz["station_names"].astype(str)[worst3_idx])
        print(f"  Auto-selected profile stations: {profile_stations}")

    # ── Plot groups ───────────────────────────────────────────────────────
    print("\n[1/5] Inversion diagnostics...")
    plot_inversion_diagnostics(npz, paths, args.dpi)

    print("\n[2/5] Spatial CV plots...")
    plot_spatial_cv(paths, npz, args.dpi)

    print("\n[3/5] Temporal CV plots...")
    plot_temporal_cv(paths, args.dpi)

    if not args.no_maps:
        print("\n[4/5] Grid map plots (loading NetCDF)...")
        ds = load_netcdf(paths["gp_nc"])
        if ds is not None:
            plot_maps(ds, npz, args.map_month, paths, args.dpi)
            print("\n[5/5] Depth-time profile heatmaps...")
            plot_profiles(ds, npz, profile_stations, paths, args.dpi)
            ds.close()
        else:
            print("  [SKIP] Maps and profiles require depth_weights_gp_grid.nc.")
    else:
        print("\n[4/5] Maps skipped (--no-maps).")
        print("[5/5] Profiles skipped (--no-maps).")

    print(f"\nAll visualisations written to: {vis_dir}")


if __name__ == "__main__":
    main()
