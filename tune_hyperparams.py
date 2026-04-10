"""
tune_hyperparams.py — Hyperparameter tuning via temporal CV (fold 3 only).

Runs a grid search over lam × lam_t using fold 3:
  Train:    months 0–71  (Jan 2015 – Dec 2020)
  Validate: months 72–82 (Jan–Nov 2021)

For each (lam, lam_t) combination, reports the per-layer and mean RMSE.
Does NOT automatically select hyperparameters — it prints a summary table
for the user to inspect and then re-run main_real.py with the chosen values.

Usage
-----
  uv run python tune_hyperparams.py [options]

Options
-------
  --data-dir PATH         Input data directory (default: my_input_data/)
  --output-dir PATH       Output directory (default: output/tune_hyperparams/)
  --sigma-insar FLOAT     InSAR noise std (default: 3.0)
  --sigma-well FLOAT      Well noise std (default: 1.0)
  --lam-candidates STR    Comma-separated lam values (default: 0.001,0.01,0.1)
  --lam-t-candidates STR  Comma-separated lam_t values (default: 0.1,0.3,1.0,3.0)
  --no-plot               Skip the heatmap figure

Outputs
-------
  output/tune_hyperparams/
    hyperparam_results_full.csv  — all combinations, per-layer RMSE columns
    hyperparam_summary_table.csv — pivot table: lam (rows) × lam_t (cols) = mean RMSE
    hyperparam_heatmap.png       — heatmap of mean RMSE (optional)
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from cv_temporal_forward import (
    compute_folds,
    compute_fold_metrics,
    load_validation_data,
    predict_validation_compaction,
    prepare_validation_arrays,
    run_training_fold,
)


def _get_fold3(data_dir: Path) -> dict:
    """
    Detect the number of epochs from the first station CSV and return
    the fold-3 definition dict produced by compute_folds().

    Parameters
    ----------
    data_dir : Path

    Returns
    -------
    dict with keys: fold, train_month_end, val_start, val_end
    """
    csv_dir = data_dir / "CSV_files"
    sample_csv = next(csv_dir.glob("*_insar_mlcw.csv"))
    df = pd.read_csv(sample_csv)
    month_cols = [c for c in df.columns if c.startswith("Month_")]
    n_epochs = len(month_cols)
    return compute_folds(n_epochs)[2]   # index 2 = fold 3


DEFAULT_LAM_CANDIDATES = [1e-3, 1e-2, 1e-1]
DEFAULT_LAM_T_CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


# ---------------------------------------------------------------------------
# Single-configuration evaluation
# ---------------------------------------------------------------------------


def evaluate_one_config(
    data_dir: Path,
    lam: float,
    lam_t: float,
    sigma_insar: float = 3.0,
    sigma_well: float = 1.0,
    fold3_def: dict | None = None,
    config_path: Path | None = None,
) -> dict:
    """
    Evaluate one (lam, lam_t) pair using fold 3 temporal CV.

    Parameters
    ----------
    data_dir : Path
    lam, lam_t, sigma_insar, sigma_well : float

    Returns
    -------
    dict with keys:
      lam, lam_t, mean_rmse_mm, max_rmse_mm, per_layer_rmse (np.ndarray),
      solver_status (str), elapsed_s (float), valid_depths_m (list[int])
    """
    t0 = time.perf_counter()

    if fold3_def is None:
        fold3_def = _get_fold3(data_dir)
    train_end = fold3_def["train_month_end"]
    val_start = fold3_def["val_start"]
    val_end   = fold3_def["val_end"]

    # ── Stage 1: training inversion ───────────────────────────────────────
    depth_weights, meta = run_training_fold(
        config_path=config_path,
        data_dir=data_dir,
        train_month_end=train_end,
        lam=lam,
        lam_t=lam_t,
        sigma_insar=sigma_insar,
        sigma_well=sigma_well,
    )
    station_names: list[str] = meta["station_names"]
    valid_depths_m: list[int] = meta["valid_depths_m"]

    # ── Load and prepare validation data ─────────────────────────────────
    insar_val_inc, mlcw_val_inc, _ = load_validation_data(
        data_dir=data_dir,
        station_names=station_names,
        valid_depths_m=valid_depths_m,
        val_month_start=val_start,
        val_month_end=val_end,
    )
    insar_val_cum, mlcw_val_cum = prepare_validation_arrays(insar_val_inc, mlcw_val_inc)

    # ── Predict and score ─────────────────────────────────────────────────
    pred = predict_validation_compaction(depth_weights, insar_val_cum)
    per_layer_df, _ = compute_fold_metrics(
        pred, mlcw_val_cum, valid_depths_m, station_names, fold_id=fold3_def["fold"]
    )

    elapsed = time.perf_counter() - t0
    per_layer_rmse = per_layer_df["rmse"].values

    return {
        "lam": lam,
        "lam_t": lam_t,
        "mean_rmse_mm": float(np.nanmean(per_layer_rmse)),
        "max_rmse_mm": float(np.nanmax(per_layer_rmse)),
        "per_layer_rmse": per_layer_rmse,
        "elapsed_s": elapsed,
        "valid_depths_m": valid_depths_m,
    }


# ---------------------------------------------------------------------------
# Results assembly
# ---------------------------------------------------------------------------


def build_results_tables(
    results: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a full results DataFrame and a lam × lam_t pivot table.

    Parameters
    ----------
    results : list of dicts from evaluate_one_config()

    Returns
    -------
    full_df    : DataFrame — one row per (lam, lam_t), plus per-layer RMSE columns
    summary_df : DataFrame — pivot table: lam (rows) × lam_t (cols) = mean RMSE
    """
    rows = []
    for r in results:
        row: dict = {
            "lam": r["lam"],
            "lam_t": r["lam_t"],
            "mean_rmse_mm": r["mean_rmse_mm"],
            "max_rmse_mm": r["max_rmse_mm"],
            "elapsed_s": r["elapsed_s"],
        }
        for l_idx, depth in enumerate(r["valid_depths_m"]):
            row[f"rmse_l{depth}"] = float(r["per_layer_rmse"][l_idx])
        rows.append(row)

    full_df = pd.DataFrame(rows)
    summary_df = full_df.pivot(index="lam", columns="lam_t", values="mean_rmse_mm")
    return full_df, summary_df


def plot_heatmap(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a heatmap of mean RMSE vs (lam, lam_t)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        summary_df,
        annot=True, fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Mean RMSE (mm)"},
    )
    ax.set_title("Fold 3 Validation: Mean RMSE vs (lam, lam_t)", fontsize=11)
    ax.set_xlabel("lam_t", fontsize=10)
    ax.set_ylabel("lam", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter grid search via fold-3 temporal CV."
    )
    p.add_argument("--data-dir", default="my_input_data/", type=Path)
    p.add_argument("--output-dir", default="output/tune_hyperparams/", type=Path)
    p.add_argument("--config", default=None, type=Path, help="Path to pipeline_config.ini")
    p.add_argument("--sigma-insar", default=3.0, type=float)
    p.add_argument("--sigma-well", default=1.0, type=float)
    p.add_argument(
        "--lam-candidates",
        default=",".join(str(v) for v in DEFAULT_LAM_CANDIDATES),
        type=str,
        help="Comma-separated lam values (e.g. '0.001,0.01,0.1').",
    )
    p.add_argument(
        "--lam-t-candidates",
        default=",".join(str(v) for v in DEFAULT_LAM_T_CANDIDATES),
        type=str,
        help="Comma-separated lam_t values (e.g. '0.1,0.3,1.0,3.0').",
    )
    p.add_argument("--no-plot", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lam_candidates = [float(v) for v in args.lam_candidates.split(",")]
    lam_t_candidates = [float(v) for v in args.lam_t_candidates.split(",")]
    n_combos = len(lam_candidates) * len(lam_t_candidates)

    fold3_def = _get_fold3(args.data_dir)
    print(
        f"Fold-3 definition: train 0–{fold3_def['train_month_end']}, "
        f"validate {fold3_def['val_start']}–{fold3_def['val_end']}"
    )

    print(
        f"Hyperparameter grid search — fold 3 only "
        f"(train 0–{fold3_def['train_month_end']}, "
        f"validate {fold3_def['val_start']}–{fold3_def['val_end']})"
    )
    print(f"lam candidates    : {lam_candidates}")
    print(f"lam_t candidates  : {lam_t_candidates}")
    print(f"Total combinations: {n_combos}")
    print()

    results = []
    t_global = time.perf_counter()

    for k, (lam, lam_t) in enumerate(product(lam_candidates, lam_t_candidates), start=1):
        print(f"  [{k:2d}/{n_combos}] lam={lam:.0e}  lam_t={lam_t:.1f} ... ", end="", flush=True)
        result = evaluate_one_config(
            data_dir=args.data_dir,
            lam=lam,
            lam_t=lam_t,
            sigma_insar=args.sigma_insar,
            sigma_well=args.sigma_well,
            fold3_def=fold3_def,
            config_path=args.config,
        )
        print(
            f"mean RMSE = {result['mean_rmse_mm']:.3f} mm  "
            f"max = {result['max_rmse_mm']:.3f} mm  "
            f"({result['elapsed_s']:.0f}s)"
        )
        results.append(result)

    total_elapsed = time.perf_counter() - t_global
    print(f"\nAll {n_combos} combinations completed in {total_elapsed:.0f}s")

    full_df, summary_df = build_results_tables(results)

    full_path = args.output_dir / "hyperparam_results_full.csv"
    summary_path = args.output_dir / "hyperparam_summary_table.csv"
    full_df.to_csv(full_path, index=False)
    summary_df.to_csv(summary_path)

    print("\n=== Mean RMSE (mm) - lam (rows) x lam_t (cols) ===")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

    best_row = full_df.loc[full_df["mean_rmse_mm"].idxmin()]
    print(f"\nBest combination: lam={best_row['lam']:.0e}  lam_t={best_row['lam_t']:.2f}  "
          f"-> mean RMSE = {best_row['mean_rmse_mm']:.4f} mm")
    print("(Inspect the table and re-run main_real.py with --lam and --lam-t accordingly.)")

    if not args.no_plot:
        try:
            plot_heatmap(summary_df, args.output_dir / "hyperparam_heatmap.png")
            print(f"\nHeatmap saved to: {args.output_dir / 'hyperparam_heatmap.png'}")
        except ImportError:
            print("  (seaborn not available; skipping heatmap)")

    print(f"\nFull results saved to: {full_path}")
    print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()
