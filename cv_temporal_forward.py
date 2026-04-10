"""
cv_temporal_forward.py — Forward chaining temporal cross-validation.

For each fold:
  Train:    months 0 .. train_end  (cumulative, CVXPY inversion)
  Validate: months val_start .. val_end

Fold definitions (0-based month indices into the sorted CSV column list):
  Fold 1: train 0–47  (Jan 2015 – Dec 2018, 48 months)  validate 48–59  (2019)
  Fold 2: train 0–59  (Jan 2015 – Dec 2019, 60 months)  validate 60–71  (2020)
  Fold 3: train 0–71  (Jan 2015 – Dec 2020, 72 months)  validate 72–82  (2021, 11 months)

For each fold the pipeline:
  1. Calls build_real_dataset(..., month_end=train_end) to get training data.
  2. Runs solve_joint_spacetime_cvxpy to get training-period depth_weights.
  3. Loads raw (incremental, NOT negated) InSAR and MLCW for the validation
     window directly from CSV files.
  4. Negates and cumsums from the start of the validation window.
  5. Predicts: pred[t, s, l] = depth_weights[s, l] * insar_val_cum[t, s]
  6. Compares against actual cumulative MLCW in the validation window.
  7. Computes per-layer RMSE, MAE, bias, correlation.

Usage
-----
  uv run python cv_temporal_forward.py [options]

Options
-------
  --data-dir PATH       Directory containing my_input_data/ (default: my_input_data/)
  --output-dir PATH     Output directory (default: output/cv_temporal/)
  --lam FLOAT           Spatial regularisation (default: 0.01)
  --lam-t FLOAT         Temporal regularisation (default: 0.3)
  --sigma-insar FLOAT   InSAR noise std (default: 3.0)
  --sigma-well FLOAT    Well noise std (default: 1.0)
  --no-plots            Skip time-series plots

Outputs
-------
  output/cv_temporal/
    temporal_cv_per_layer.csv       — RMSE per fold per layer (primary result)
    temporal_cv_per_station_layer.csv — per fold, station, layer breakdown
    temporal_cv_summary.csv         — summary per fold
    temporal_cv_arrays.npz          — full pred/true arrays
    plots/fold{N}_station_{name}.png — time series (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import build_real_dataset, parse_dataset_config
from src.solvers_temporal import solve_joint_spacetime_cvxpy


# ---------------------------------------------------------------------------
# Fold definitions
# ---------------------------------------------------------------------------

def compute_folds(n_epochs: int) -> list[dict]:
    """
    Compute forward-chaining fold boundaries from the total number of epochs.

    Uses a three-fold scheme: 60% / 20% / 20% split (approximate), with a
    minimum of 6 epochs per fold.  The validation windows are non-overlapping
    and contiguous.

    Parameters
    ----------
    n_epochs : int
        Total number of monthly epochs in the dataset.

    Returns
    -------
    list of dict, each with keys:
        fold (int), train_month_end (int), val_start (int), val_end (int)
    """
    if n_epochs < 12:
        raise ValueError(
            f"Dataset has only {n_epochs} epochs. Need at least 12 for 3-fold CV."
        )
    # Each validation window is ~13% of total epochs, minimum 6 months.
    val_len = max(6, n_epochs // 8)
    # Fold 3: last val_len epochs
    f3_end = n_epochs - 1
    f3_start = f3_end - val_len + 1
    # Fold 2: immediately before fold 3
    f2_end = f3_start - 1
    f2_start = f2_end - val_len + 1
    # Fold 1: immediately before fold 2
    f1_end = f2_start - 1
    f1_start = f1_end - val_len + 1
    if f1_start < 1:
        raise ValueError(
            f"Dataset has only {n_epochs} epochs. Cannot build 3 non-overlapping "
            "folds with sufficient training data. Reduce val_len or add more epochs."
        )
    return [
        {"fold": 1, "train_month_end": f1_start - 1, "val_start": f1_start, "val_end": f1_end},
        {"fold": 2, "train_month_end": f2_start - 1, "val_start": f2_start, "val_end": f2_end},
        {"fold": 3, "train_month_end": f3_start - 1, "val_start": f3_start, "val_end": f3_end},
    ]


# Module-level fallback for backward compatibility (84-epoch real dataset).
# Replaced at runtime inside main() by compute_folds(n_epochs).
FOLDS = compute_folds(84)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_depth_weights(
    m_flat: np.ndarray,
    d_insar_flat: np.ndarray,
    n_epochs: int,
    n_stations: int,
    n_layers: int,
) -> np.ndarray:
    """
    Compute depth_weights from a solver output.  Mirrors main_real.py.

    Returns
    -------
    depth_weights : np.ndarray, shape (n_stations, n_layers)
    """
    m_est = m_flat.reshape(n_epochs, n_stations, n_layers)
    cumulative = m_est.sum(axis=0)                                    # (n_stations, n_layers)
    insar_cum = d_insar_flat.reshape(n_epochs, n_stations).sum(axis=0)  # (n_stations,)
    insar_safe = np.where(insar_cum > 0, insar_cum, 1.0)
    return cumulative / insar_safe[:, np.newaxis]


def run_training_fold(
    config_path: Path | None,
    data_dir: Path,
    train_month_end: int,
    lam: float,
    lam_t: float,
    sigma_insar: float = 3.0,
    sigma_well: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Load training data and run inversion for one fold.

    Parameters
    ----------
    data_dir : Path
    train_month_end : int
        0-based inclusive end index for training months.
    lam, lam_t, sigma_insar, sigma_well : float

    Returns
    -------
    depth_weights_train : np.ndarray, shape (n_stations, n_layers)
    meta : dict  (includes station_names, valid_depths_m, x_twd97, y_twd97, ...)
    """
    dataset_kwargs = parse_dataset_config(config_path)
    dataset, config, meta = build_real_dataset(
        **dataset_kwargs,
        data_dir=data_dir,
        month_start=0,
        month_end=train_month_end,
        lam=lam,
        lam_t=lam_t,
        sigma_insar=sigma_insar,
        sigma_well=sigma_well,
        cumulate=True,
    )
    m_flat = solve_joint_spacetime_cvxpy(
        dataset,
        x_twd97=meta["x_twd97"],
        y_twd97=meta["y_twd97"],
    )
    depth_weights = compute_depth_weights(
        m_flat,
        dataset.d_insar,
        config.n_epochs,
        config.n_pixels,
        config.n_layers,
    )
    return depth_weights, meta


# ---------------------------------------------------------------------------
# Validation data loading (raw, directly from CSVs)
# ---------------------------------------------------------------------------


def load_validation_data(
    data_dir: Path,
    station_names: list[str],
    valid_depths_m: list[int],
    val_month_start: int,
    val_month_end: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load raw incremental InSAR and MLCW for the validation window.

    Data are NOT negated and NOT cumsummed.  NaN is used for missing entries.

    Parameters
    ----------
    data_dir : Path
    station_names : list[str]
    valid_depths_m : list[int]  — MLCW depth levels in metres (excluding 0 and 300)
    val_month_start, val_month_end : int — 0-based inclusive indices

    Returns
    -------
    insar_val_inc  : np.ndarray, shape (n_val, n_stations)   — raw, NaN for missing
    mlcw_val_inc   : np.ndarray, shape (n_val, n_stations, n_layers) — raw, NaN for missing
    val_month_labels : list[str]  — e.g. ['Month_049', ..., 'Month_060']
    """
    csv_dir = data_dir / "CSV_files"

    # Determine month column names for the validation window
    # Convention: 0-based index k → 'Month_{k+1:03d}'
    val_month_labels = [
        f"Month_{k + 1:03d}" for k in range(val_month_start, val_month_end + 1)
    ]
    n_val = len(val_month_labels)
    n_stations = len(station_names)
    n_layers = len(valid_depths_m)

    insar_val_inc = np.full((n_val, n_stations), np.nan)
    mlcw_val_inc = np.full((n_val, n_stations, n_layers), np.nan)

    for s_idx, name in enumerate(station_names):
        csv_path = csv_dir / f"{name}_insar_mlcw.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).set_index("Depth")

        # InSAR surface (depth = 0)
        if 0 in df.index:
            for t_idx, mc in enumerate(val_month_labels):
                if mc in df.columns:
                    val = df.loc[0, mc]
                    if not pd.isna(val):
                        insar_val_inc[t_idx, s_idx] = float(val)

        # MLCW layers
        for l_idx, depth in enumerate(valid_depths_m):
            if depth not in df.index:
                continue
            for t_idx, mc in enumerate(val_month_labels):
                if mc in df.columns:
                    val = df.loc[depth, mc]
                    if not pd.isna(val):
                        mlcw_val_inc[t_idx, s_idx, l_idx] = float(val)

    return insar_val_inc, mlcw_val_inc, val_month_labels


def prepare_validation_arrays(
    insar_val_inc: np.ndarray,
    mlcw_val_inc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply sign negation and cumsum from the start of the validation window.

    The cumsum origin is reset to zero at the first validation month, so
    both arrays represent displacement *accumulated within the validation
    window only*.  This makes the units consistent with depth_weights, which
    are derived from the same sign convention (positive = compaction).

    Parameters
    ----------
    insar_val_inc : (n_val, n_stations)
    mlcw_val_inc  : (n_val, n_stations, n_layers)

    Returns
    -------
    insar_val_cum : (n_val, n_stations)
    mlcw_val_cum  : (n_val, n_stations, n_layers)
    """
    # Negate: subsidence is negative in raw data, positive in code convention
    insar_neg = -insar_val_inc
    mlcw_neg = -mlcw_val_inc

    # Cumsum from start of validation window; NaN in place of all-NaN series
    insar_cum = np.nancumsum(insar_neg, axis=0)
    mlcw_cum = np.nancumsum(mlcw_neg, axis=0)

    # Restore NaN where the original data had no valid values throughout
    all_nan_insar = np.isnan(insar_val_inc).all(axis=0)     # (n_stations,)
    insar_cum[:, all_nan_insar] = np.nan

    all_nan_mlcw = np.isnan(mlcw_val_inc).all(axis=0)       # (n_stations, n_layers)
    for t in range(mlcw_cum.shape[0]):
        mlcw_cum[t][all_nan_mlcw] = np.nan

    return insar_cum, mlcw_cum


# ---------------------------------------------------------------------------
# Prediction and metrics
# ---------------------------------------------------------------------------


def predict_validation_compaction(
    depth_weights: np.ndarray,
    insar_val_cum: np.ndarray,
) -> np.ndarray:
    """
    Predict cumulative layer compaction for the validation window.

    pred[t, s, l] = depth_weights[s, l] * insar_val_cum[t, s]

    Parameters
    ----------
    depth_weights : (n_stations, n_layers)
    insar_val_cum : (n_val, n_stations)

    Returns
    -------
    pred : np.ndarray, shape (n_val, n_stations, n_layers)
    """
    return depth_weights[np.newaxis, :, :] * insar_val_cum[:, :, np.newaxis]


def compute_fold_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    valid_depths_m: list[int],
    station_names: list[str],
    fold_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-layer and per-station-layer metrics for one fold.

    Parameters
    ----------
    pred : (n_val, n_stations, n_layers) — predicted cumulative compaction (mm)
    true : (n_val, n_stations, n_layers) — observed cumulative MLCW (mm)
    valid_depths_m : list[int], length n_layers
    station_names : list[str], length n_stations
    fold_id : int

    Returns
    -------
    per_layer_df : pd.DataFrame — columns: fold, depth_m, rmse, mae, bias, correlation, n_valid
    per_station_df : pd.DataFrame — columns: fold, station, depth_m, rmse, mae, n_valid
    """
    from scipy.stats import pearsonr

    n_layers = len(valid_depths_m)
    n_stations = len(station_names)

    per_layer_rows = []
    per_station_rows = []

    for l_idx, depth in enumerate(valid_depths_m):
        pred_l = pred[:, :, l_idx]    # (n_val, n_stations)
        true_l = true[:, :, l_idx]    # (n_val, n_stations)

        # Flatten and mask NaN
        valid = ~np.isnan(true_l) & ~np.isnan(pred_l)
        p_flat = pred_l[valid]
        t_flat = true_l[valid]
        n_valid = int(valid.sum())

        if n_valid < 2:
            corr = np.nan
            rmse = np.nan
            mae = np.nan
            bias = np.nan
        else:
            err = p_flat - t_flat
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            bias = float(np.mean(err))
            try:
                corr = float(pearsonr(p_flat, t_flat)[0])
            except Exception:
                corr = np.nan

        per_layer_rows.append({
            "fold": fold_id,
            "depth_m": depth,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "correlation": corr,
            "n_valid": n_valid,
        })

        # Per-station breakdown for this layer
        for s_idx, sname in enumerate(station_names):
            p_s = pred_l[:, s_idx]
            t_s = true_l[:, s_idx]
            v_s = ~np.isnan(t_s) & ~np.isnan(p_s)
            nv = int(v_s.sum())
            if nv < 1:
                s_rmse = np.nan
                s_mae = np.nan
            else:
                e_s = p_s[v_s] - t_s[v_s]
                s_rmse = float(np.sqrt(np.mean(e_s**2)))
                s_mae = float(np.mean(np.abs(e_s)))
            per_station_rows.append({
                "fold": fold_id,
                "station": sname,
                "depth_m": depth,
                "rmse": s_rmse,
                "mae": s_mae,
                "n_valid": nv,
            })

    return pd.DataFrame(per_layer_rows), pd.DataFrame(per_station_rows)


def plot_fold_timeseries(
    pred: np.ndarray,
    true: np.ndarray,
    insar_val_cum: np.ndarray,
    station_names: list[str],
    valid_depths_m: list[int],
    fold_id: int,
    val_month_start: int,
    output_dir: Path,
) -> None:
    """
    Save per-station time-series comparison plots for one fold.

    For each station: a figure with n_layers subplots (predicted vs observed).
    Saves as output_dir/fold{fold_id}_station_{name}.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_val, n_stations, n_layers = pred.shape
    x_axis = np.arange(val_month_start, val_month_start + n_val)

    for s_idx, sname in enumerate(station_names):
        n_cols = 5
        n_rows = int(np.ceil(n_layers / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
        fig.suptitle(f"Fold {fold_id} — Station {sname}", fontsize=12)

        for l_idx, depth in enumerate(valid_depths_m):
            ax = axes[l_idx // n_cols][l_idx % n_cols]
            ax.plot(x_axis, pred[:, s_idx, l_idx], label="Predicted", linewidth=1.2)
            ax.plot(x_axis, true[:, s_idx, l_idx], label="Observed (MLCW)",
                    linestyle="--", linewidth=1.2)
            ax.set_title(f"{depth} m", fontsize=8)
            ax.tick_params(labelsize=7)
            if l_idx == 0:
                ax.legend(fontsize=7)

        # Hide unused subplots
        total_cells = n_rows * n_cols
        for empty in range(n_layers, total_cells):
            axes[empty // n_cols][empty % n_cols].set_visible(False)

        plt.tight_layout()
        fname = output_dir / f"fold{fold_id}_station_{sname}.png"
        fig.savefig(fname, dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward chaining temporal cross-validation.")
    p.add_argument("--data-dir", default="my_input_data/", type=Path)
    p.add_argument("--output-dir", default="output/cv_temporal/", type=Path)
    p.add_argument("--config", default=None, type=Path, help="Path to pipeline_config.ini")
    p.add_argument("--lam", default=0.01, type=float)
    p.add_argument("--lam-t", default=0.3, type=float)
    p.add_argument("--sigma-insar", default=3.0, type=float)
    p.add_argument("--sigma-well", default=1.0, type=float)
    p.add_argument("--no-plots", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect number of epochs and recompute fold boundaries.
    _sample_csv = next((args.data_dir / "CSV_files").glob("*_insar_mlcw.csv"))
    _sample_df = pd.read_csv(_sample_csv)
    _n_epochs = len([c for c in _sample_df.columns if c.startswith("Month_")])
    folds = compute_folds(_n_epochs)
    print(f"Detected {_n_epochs} epochs. Fold boundaries: {folds}")

    all_per_layer: list[pd.DataFrame] = []
    all_per_station: list[pd.DataFrame] = []
    fold_summaries: list[dict] = []

    # Arrays for NPZ (pad to max val length with NaN)
    max_val_len = max(f["val_end"] - f["val_start"] + 1 for f in folds)

    pred_store: list[np.ndarray] = []
    true_store: list[np.ndarray] = []

    for fold_def in folds:
        fold_id = fold_def["fold"]
        train_end = fold_def["train_month_end"]
        val_start = fold_def["val_start"]
        val_end = fold_def["val_end"]
        n_val = val_end - val_start + 1

        print(f"\n{'='*60}")
        print(f"Fold {fold_id}: train months 0–{train_end}, validate {val_start}–{val_end}")
        print(f"{'='*60}")

        # ── Stage 1: train inversion ───────────────────────────────────────
        print("  Running Stage 1 inversion on training period ...")
        depth_weights, meta = run_training_fold(
            config_path=args.config,
            data_dir=args.data_dir,
            train_month_end=train_end,
            lam=args.lam,
            lam_t=args.lam_t,
            sigma_insar=args.sigma_insar,
            sigma_well=args.sigma_well,
        )
        station_names: list[str] = meta["station_names"]
        valid_depths_m: list[int] = meta["valid_depths_m"]
        n_stations = len(station_names)
        n_layers = len(valid_depths_m)
        print(f"  Stations: {n_stations}, Layers: {n_layers}")
        print(f"  Mean depth_weights sum (across layers): "
              f"{depth_weights.sum(axis=1).mean():.3f}")

        # ── Load raw validation data ───────────────────────────────────────
        print(f"  Loading validation data (months {val_start}–{val_end}) ...")
        insar_val_inc, mlcw_val_inc, val_labels = load_validation_data(
            data_dir=args.data_dir,
            station_names=station_names,
            valid_depths_m=valid_depths_m,
            val_month_start=val_start,
            val_month_end=val_end,
        )
        # Fraction of non-NaN entries
        valid_frac_insar = float((~np.isnan(insar_val_inc)).mean())
        valid_frac_mlcw = float((~np.isnan(mlcw_val_inc)).mean())
        print(f"  InSAR valid: {valid_frac_insar*100:.1f}%  "
              f"MLCW valid: {valid_frac_mlcw*100:.1f}%")

        # ── Cumsum from validation window start ────────────────────────────
        insar_val_cum, mlcw_val_cum = prepare_validation_arrays(
            insar_val_inc, mlcw_val_inc
        )

        # ── Predict ────────────────────────────────────────────────────────
        pred = predict_validation_compaction(depth_weights, insar_val_cum)
        # pred: (n_val, n_stations, n_layers)

        # ── Metrics ────────────────────────────────────────────────────────
        per_layer_df, per_station_df = compute_fold_metrics(
            pred, mlcw_val_cum, valid_depths_m, station_names, fold_id
        )
        mean_rmse = per_layer_df["rmse"].mean()
        max_rmse = per_layer_df["rmse"].max()
        print(f"  Mean RMSE (all layers): {mean_rmse:.3f} mm")
        print(f"  Max  RMSE (all layers): {max_rmse:.3f} mm")

        all_per_layer.append(per_layer_df)
        all_per_station.append(per_station_df)
        fold_summaries.append({
            "fold": fold_id,
            "train_month_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "mean_rmse_mm": mean_rmse,
            "max_rmse_mm": max_rmse,
            "n_stations": n_stations,
        })

        # ── Store for NPZ (pad with NaN to max_val_len) ───────────────────
        def _pad(arr: np.ndarray, target_len: int) -> np.ndarray:
            """Pad time axis to target_len with NaN."""
            pad_n = target_len - arr.shape[0]
            if pad_n == 0:
                return arr
            pad_shape = (pad_n,) + arr.shape[1:]
            return np.concatenate([arr, np.full(pad_shape, np.nan)], axis=0)

        pred_store.append(_pad(pred, max_val_len))
        true_store.append(_pad(mlcw_val_cum, max_val_len))

        # ── Plots ──────────────────────────────────────────────────────────
        if not args.no_plots:
            print("  Saving time-series plots ...")
            plot_fold_timeseries(
                pred=pred,
                true=mlcw_val_cum,
                insar_val_cum=insar_val_cum,
                station_names=station_names,
                valid_depths_m=valid_depths_m,
                fold_id=fold_id,
                val_month_start=val_start,
                output_dir=plots_dir,
            )

    # ── Aggregate and save ────────────────────────────────────────────────
    all_per_layer_df = pd.concat(all_per_layer, ignore_index=True)
    all_per_station_df = pd.concat(all_per_station, ignore_index=True)
    summary_df = pd.DataFrame(fold_summaries)

    per_layer_path = args.output_dir / "temporal_cv_per_layer.csv"
    per_station_path = args.output_dir / "temporal_cv_per_station_layer.csv"
    summary_path = args.output_dir / "temporal_cv_summary.csv"

    all_per_layer_df.to_csv(per_layer_path, index=False)
    all_per_station_df.to_csv(per_station_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    np.savez(
        args.output_dir / "temporal_cv_arrays.npz",
        pred=np.stack(pred_store, axis=0),   # (3, max_val_len, n_stations, n_layers)
        true=np.stack(true_store, axis=0),
        valid_depths_m=np.array(valid_depths_m),
        station_names=np.array(station_names),
    )

    print("\n" + "="*60)
    print("Temporal CV Summary")
    print("="*60)
    print(summary_df.to_string(index=False))
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  {per_layer_path.name}")
    print(f"  {per_station_path.name}")
    print(f"  {summary_path.name}")
    print("  temporal_cv_arrays.npz")


if __name__ == "__main__":
    main()
