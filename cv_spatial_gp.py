"""
cv_spatial_gp.py — Leave-One-Station-Out (LOSO) spatial cross-validation
for the GP depth-weight interpolation.

For each station i (0..n_stations-1):
  1. Remove station i's data from the full dataset (post-hoc array surgery).
  2. Run Stage 1 inversion (CVXPY) on the remaining n-1 stations.
  3. Fit per-layer GPs on the n-1 station weights.
  4. Predict depth_weights at station i's coordinates.
  5. Compare against the reference weights from the full-data inversion.

The full dataset is loaded ONCE. Each fold operates via array surgery to avoid
re-running the I/O-heavy loader per fold, and to preserve the month column
range (month column auto-detection uses station_names[0], which must not
change per fold).

Usage
-----
  uv run python cv_spatial_gp.py [options]

Options
-------
  --data-dir PATH       Directory containing my_input_data/ (default: my_input_data/)
  --inversion-npz PATH  Full-data inversion result (default: output/real_m_est.npz)
  --output-dir PATH     Output directory (default: output/cv_spatial_gp/)
  --lam FLOAT           Spatial regularisation (default: 0.01)
  --lam-t FLOAT         Temporal regularisation (default: 0.3)
  --n-restarts INT      GP optimiser restarts per layer (default: 10)
  --sigma-insar FLOAT   InSAR noise std (default: 3.0)
  --sigma-well FLOAT    Well noise std (default: 1.0)

Outputs
-------
  output/cv_spatial_gp/
    loso_per_layer.csv         — RMSE/MAE/bias per depth layer (primary CV result)
    loso_per_station.csv       — error per station per layer
    loso_gp_predictions.npz   — full arrays: mean_pred, std_pred, true_weights
    loso_summary.txt           — plain-text summary
"""


import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

from src.config import SyntheticDataset, SystemConfig
from src.loader import build_real_dataset, parse_dataset_config
from src.solvers_temporal import solve_joint_spacetime_cvxpy


# ---------------------------------------------------------------------------
# Dataset subsetting (post-hoc array surgery)
# ---------------------------------------------------------------------------


def subset_dataset_exclude_station(
    full_dataset: SyntheticDataset,
    full_config: SystemConfig,
    full_meta: Dict[str, Any],
    exclude_idx: int,
) -> Tuple[SyntheticDataset, SystemConfig, Dict[str, Any]]:
    """
    Return a new (dataset, config, meta) with station ``exclude_idx`` removed.

    No I/O is performed; this is pure array surgery on the already-loaded
    full dataset.

    Parameters
    ----------
    full_dataset : SyntheticDataset
        Full-dataset result from build_real_dataset(cumulate=True).
    full_config : SystemConfig
        Corresponding config (grid_rows = n_stations, grid_cols = 1).
    full_meta : dict
        Corresponding metadata dict.
    exclude_idx : int
        0-based station index to remove.

    Returns
    -------
    sub_dataset : SyntheticDataset
    sub_config  : SystemConfig
    sub_meta    : dict
    """
    n_stations = full_config.n_pixels
    n_layers = full_config.n_layers
    n_epochs = full_config.n_epochs

    # ── d_insar surgery ───────────────────────────────────────────────────
    # Layout: epoch-major → (n_epochs, n_stations)
    d_insar_2d = full_dataset.d_insar.reshape(n_epochs, n_stations)
    mask_stations = np.ones(n_stations, dtype=bool)
    mask_stations[exclude_idx] = False
    d_insar_sub = d_insar_2d[:, mask_stations].flatten()

    # ── w surgery ─────────────────────────────────────────────────────────
    # Layout: epoch-major, within each epoch station-major
    # (n_epochs, n_stations * n_layers)
    w_2d = full_dataset.w.reshape(n_epochs, n_stations * n_layers)
    # Build column mask: True for layers NOT belonging to exclude_idx
    col_mask = np.ones(n_stations * n_layers, dtype=bool)
    col_start = exclude_idx * n_layers
    col_mask[col_start : col_start + n_layers] = False
    w_sub = w_2d[:, col_mask].flatten()

    n_sub = n_stations - 1

    # ── New config ────────────────────────────────────────────────────────
    sub_config = SystemConfig(
        grid_rows=n_sub,
        grid_cols=1,
        n_layers=n_layers,
        well_indices=tuple(range(n_sub)),
        delta_z=full_config.delta_z,
        n_epochs=n_epochs,
        lam=full_config.lam,
        lam_t=full_config.lam_t,
        sigma_insar=full_config.sigma_insar,
        sigma_well=full_config.sigma_well,
        spatial_dist_threshold_m=full_config.spatial_dist_threshold_m,
        cumulative_space=full_config.cumulative_space,
    )

    sub_dataset = SyntheticDataset(
        config=sub_config,
        m_true=np.zeros(sub_config.total_voxels, dtype=np.float64),
        d_insar=d_insar_sub.astype(np.float64),
        w=w_sub.astype(np.float64),
    )

    # ── New meta ──────────────────────────────────────────────────────────
    sub_meta = dict(full_meta)
    sub_meta["station_names"] = [
        s for k, s in enumerate(full_meta["station_names"]) if k != exclude_idx
    ]
    sub_meta["x_twd97"] = [
        v for k, v in enumerate(full_meta["x_twd97"]) if k != exclude_idx
    ]
    sub_meta["y_twd97"] = [
        v for k, v in enumerate(full_meta["y_twd97"]) if k != exclude_idx
    ]

    return sub_dataset, sub_config, sub_meta


# ---------------------------------------------------------------------------
# Depth weight computation (mirrors main_real.py logic)
# ---------------------------------------------------------------------------


def compute_depth_weights(
    m_flat: np.ndarray,
    d_insar_flat: np.ndarray,
    n_epochs: int,
    n_stations: int,
    n_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute depth_weights and insar_coverage from a solver output.

    Mirrors the main_real.py post-processing exactly.

    Parameters
    ----------
    m_flat : np.ndarray, shape (n_epochs * n_stations * n_layers,)
    d_insar_flat : np.ndarray, shape (n_epochs * n_stations,)
    n_epochs, n_stations, n_layers : int

    Returns
    -------
    depth_weights : np.ndarray, shape (n_stations, n_layers)
    insar_coverage : np.ndarray, shape (n_stations,)
    """
    m_est = m_flat.reshape(n_epochs, n_stations, n_layers)
    cumulative = m_est.sum(axis=0)  # (n_stations, n_layers)

    insar_cum = d_insar_flat.reshape(n_epochs, n_stations).sum(axis=0)  # (n_stations,)
    insar_safe = np.where(insar_cum > 0, insar_cum, 1.0)

    depth_weights = cumulative / insar_safe[:, np.newaxis]
    insar_coverage = cumulative.sum(axis=1) / insar_safe
    return depth_weights, insar_coverage


# ---------------------------------------------------------------------------
# Single LOSO fold
# ---------------------------------------------------------------------------


def run_loso_fold(
    full_dataset: SyntheticDataset,
    full_config: SystemConfig,
    full_meta: Dict[str, Any],
    full_depth_weights: np.ndarray,
    exclude_idx: int,
    n_gp_restarts: int = 10,
) -> Dict[str, Any]:
    """
    Run one LOSO fold for station ``exclude_idx``.

    Parameters
    ----------
    full_dataset : SyntheticDataset
    full_config  : SystemConfig
    full_meta    : dict
    full_depth_weights : np.ndarray, shape (n_stations, n_layers)
        Reference weights from the full-data inversion.
    exclude_idx  : int
    n_gp_restarts : int

    Returns
    -------
    dict with keys:
      station_name, exclude_idx,
      mean_pred (n_layers,), std_pred (n_layers,),
      true_weight (n_layers,),
      error (n_layers,),       # mean_pred - true_weight
      insar_coverage_sub,      # float — coverage at the held-out station
                               # computed from the sub-inversion (diagnostic)
    """
    from stage2_gp_interpolation import fit_gp_per_layer

    n_epochs = full_config.n_epochs
    n_layers = full_config.n_layers

    # ── Build n-1 station dataset ─────────────────────────────────────────
    sub_dataset, sub_config, sub_meta = subset_dataset_exclude_station(
        full_dataset, full_config, full_meta, exclude_idx
    )
    n_sub = sub_config.n_pixels

    # ── Run Stage 1 on n-1 stations ───────────────────────────────────────
    m_flat_sub = solve_joint_spacetime_cvxpy(
        sub_dataset,
        x_twd97=sub_meta["x_twd97"],
        y_twd97=sub_meta["y_twd97"],
    )

    depth_weights_sub, _ = compute_depth_weights(
        m_flat_sub, sub_dataset.d_insar, n_epochs, n_sub, n_layers
    )

    # ── GP: train on n-1, predict at excluded station ─────────────────────
    x_train_km = np.array(sub_meta["x_twd97"]) / 1000.0
    y_train_km = np.array(sub_meta["y_twd97"]) / 1000.0
    x_pred_km = np.array([full_meta["x_twd97"][exclude_idx]]) / 1000.0
    y_pred_km = np.array([full_meta["y_twd97"][exclude_idx]]) / 1000.0

    mean_pred_2d, std_pred_2d, _ = fit_gp_per_layer(
        depth_weights=depth_weights_sub,
        x_train_km=x_train_km,
        y_train_km=y_train_km,
        x_pred_km=x_pred_km,
        y_pred_km=y_pred_km,
        n_restarts=n_gp_restarts,
    )
    # mean_pred_2d: (n_layers, 1); flatten to (n_layers,)
    mean_pred = mean_pred_2d[:, 0]
    std_pred = std_pred_2d[:, 0]

    true_weight = full_depth_weights[exclude_idx, :]  # (n_layers,)

    return {
        "station_name": full_meta["station_names"][exclude_idx],
        "exclude_idx": exclude_idx,
        "mean_pred": mean_pred,
        "std_pred": std_pred,
        "true_weight": true_weight,
        "error": mean_pred - true_weight,
    }


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def compute_loso_metrics(
    all_folds: List[Dict[str, Any]],
    valid_depths_m: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate LOSO fold results into per-station and per-layer metric tables.

    Parameters
    ----------
    all_folds : list of dicts from run_loso_fold()
    valid_depths_m : list of int, length n_layers

    Returns
    -------
    per_station_df : pd.DataFrame
        One row per station. Columns: station_name, total_col_rmse,
        total_col_mae, low_reliability, plus error_l{depth} for each layer.
    per_layer_df : pd.DataFrame
        One row per layer. Columns: depth_m, rmse, mae, bias, std_error.
        This is the primary input to the uncertainty update.
    """
    n_stations = len(all_folds)
    n_layers = len(valid_depths_m)

    errors = np.stack([f["error"] for f in all_folds], axis=0)       # (n_stations, n_layers)
    mean_preds = np.stack([f["mean_pred"] for f in all_folds], axis=0)
    true_weights = np.stack([f["true_weight"] for f in all_folds], axis=0)

    # Per-layer metrics
    rmse_per_layer = np.sqrt(np.mean(errors**2, axis=0))     # (n_layers,)
    mae_per_layer = np.mean(np.abs(errors), axis=0)
    bias_per_layer = np.mean(errors, axis=0)
    std_per_layer = np.std(errors, axis=0)

    per_layer_df = pd.DataFrame({
        "depth_m": valid_depths_m,
        "rmse": rmse_per_layer,
        "mae": mae_per_layer,
        "bias": bias_per_layer,
        "std_error": std_per_layer,
    })

    # Per-station metrics
    total_col_pred = mean_preds.sum(axis=1)   # (n_stations,)
    total_col_true = true_weights.sum(axis=1)
    total_col_error = total_col_pred - total_col_true

    rows = []
    for k, fold in enumerate(all_folds):
        row = {
            "station_name": fold["station_name"],
            "total_col_pred": float(total_col_pred[k]),
            "total_col_true": float(total_col_true[k]),
            "total_col_error": float(total_col_error[k]),
            "total_col_abs_error": float(abs(total_col_error[k])),
        }
        for l_idx, d in enumerate(valid_depths_m):
            row[f"error_l{d}"] = float(errors[k, l_idx])
        rows.append(row)

    per_station_df = pd.DataFrame(rows)
    return per_station_df, per_layer_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOSO spatial cross-validation for GP interpolation.")
    p.add_argument("--data-dir", default="my_input_data/", type=Path)
    p.add_argument("--inversion-npz", default="output/real_m_est.npz", type=Path)
    p.add_argument("--output-dir", default="output/cv_spatial_gp/", type=Path)
    p.add_argument("--config", default=None, type=Path, help="Path to pipeline_config.ini")
    p.add_argument("--lam", default=0.01, type=float)
    p.add_argument("--lam-t", default=0.3, type=float)
    p.add_argument("--n-restarts", default=10, type=int)
    p.add_argument("--sigma-insar", default=3.0, type=float)
    p.add_argument("--sigma-well", default=1.0, type=float)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load full dataset (once) ──────────────────────────────────────────
    print(f"Loading full dataset from: {args.data_dir}")
    full_dataset, full_config, full_meta = build_real_dataset(
        data_dir=args.data_dir,
        month_start=0,
        month_end=None,
        lam=args.lam,
        lam_t=args.lam_t,
        sigma_insar=args.sigma_insar,
        sigma_well=args.sigma_well,
        cumulate=True,
    )
    n_stations = full_config.n_pixels
    n_layers = full_config.n_layers
    n_epochs = full_config.n_epochs
    valid_depths_m: List[int] = full_meta["valid_depths_m"]
    station_names: List[str] = full_meta["station_names"]
    print(f"  Stations: {n_stations}, Layers: {n_layers}, Epochs: {n_epochs}")

    # ── Load reference depth_weights from full-data inversion ─────────────
    print(f"\nLoading full-data reference weights from: {args.inversion_npz}")
    ref = np.load(args.inversion_npz, allow_pickle=True)
    full_depth_weights = ref["depth_weights"]   # (n_stations, n_layers)
    if full_depth_weights.shape != (n_stations, n_layers):
        raise ValueError(
            f"NPZ depth_weights shape {full_depth_weights.shape} does not match "
            f"loaded dataset ({n_stations}, {n_layers}). "
            "Run main_real.py first with the same data_dir."
        )

    # ── Run LOSO folds ────────────────────────────────────────────────────
    all_folds = []
    print(f"\nRunning {n_stations} LOSO folds ...")
    t_total = time.perf_counter()

    for i in range(n_stations):
        name = station_names[i]
        print(f"  Fold {i+1:2d}/{n_stations}: excluding {name} ... ", end="", flush=True)
        t0 = time.perf_counter()
        fold_result = run_loso_fold(
            full_dataset=full_dataset,
            full_config=full_config,
            full_meta=full_meta,
            full_depth_weights=full_depth_weights,
            exclude_idx=i,
            n_gp_restarts=args.n_restarts,
        )
        elapsed = time.perf_counter() - t0
        total_col_rmse = float(np.abs(fold_result["error"].sum()))
        print(f"done ({elapsed:.0f}s)  total-col error = {total_col_rmse:.4f}")
        all_folds.append(fold_result)

    total_elapsed = time.perf_counter() - t_total
    print(f"\nAll folds completed in {total_elapsed:.0f}s")

    # ── Compute metrics ───────────────────────────────────────────────────
    per_station_df, per_layer_df = compute_loso_metrics(all_folds, valid_depths_m)

    # ── Save outputs ──────────────────────────────────────────────────────
    per_layer_path = args.output_dir / "loso_per_layer.csv"
    per_station_path = args.output_dir / "loso_per_station.csv"
    per_layer_df.to_csv(per_layer_path, index=False)
    per_station_df.to_csv(per_station_path, index=False)

    # NPZ with full arrays
    mean_pred_all = np.stack([f["mean_pred"] for f in all_folds], axis=0)
    std_pred_all = np.stack([f["std_pred"] for f in all_folds], axis=0)
    error_all = np.stack([f["error"] for f in all_folds], axis=0)
    np.savez(
        args.output_dir / "loso_gp_predictions.npz",
        mean_pred=mean_pred_all,
        std_pred=std_pred_all,
        true_weights=full_depth_weights,
        error=error_all,
        station_names=np.array(station_names),
        valid_depths_m=np.array(valid_depths_m),
    )

    # Summary text
    mean_rmse = per_layer_df["rmse"].mean()
    max_rmse = per_layer_df["rmse"].max()
    worst_layer = per_layer_df.loc[per_layer_df["rmse"].idxmax(), "depth_m"]
    summary_lines = [
        "=== LOSO Spatial CV Summary ===",
        f"Stations: {n_stations}  Layers: {n_layers}  Epochs: {n_epochs}",
        f"lam={args.lam}  lam_t={args.lam_t}",
        "",
        f"Mean RMSE across layers (weight units): {mean_rmse:.5f}",
        f"Max  RMSE across layers (weight units): {max_rmse:.5f}  at depth {worst_layer} m",
        "",
        "Per-layer RMSE (top 5 worst):",
    ]
    top5 = per_layer_df.nlargest(5, "rmse")[["depth_m", "rmse", "mae", "bias"]]
    summary_lines.append(top5.to_string(index=False))
    summary_lines += [
        "",
        "Per-station total-column abs error (top 5 worst):",
    ]
    top5s = per_station_df.nlargest(5, "total_col_abs_error")[
        ["station_name", "total_col_error", "total_col_abs_error"]
    ]
    summary_lines.append(top5s.to_string(index=False))

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (args.output_dir / "loso_summary.txt").write_text(summary_text)

    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  {per_layer_path.name}")
    print(f"  {per_station_path.name}")
    print("  loso_gp_predictions.npz")
    print("  loso_summary.txt")


if __name__ == "__main__":
    main()
