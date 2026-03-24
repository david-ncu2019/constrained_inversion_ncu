"""
main_real.py — Joint space-time inversion using real CRFP monitoring data.

Usage
-----
  uv run python main_real.py [options]

Options
-------
  --data-dir PATH       Directory containing my_input_data/ structure.
                        Default: my_input_data/
  --month-start INT     First Month_N to include (default: 0 = Jan 2015).
  --month-end   INT     Last Month_N to include  (default: 83 = Dec 2021).
  --lam   FLOAT         Spatial regularisation parameter (default: 1e-2).
  --lam-t FLOAT         Temporal regularisation parameter (default: 1.0).
  --sigma-insar FLOAT   InSAR noise std dev in mm (default: 3.0).
  --sigma-well  FLOAT   Well noise std dev in mm  (default: 1.0).
  --solver {joint,independent}
                        'joint'       : solve_joint_spacetime (recommended).
                        'independent' : solve_independent_epochs (faster, no
                                        temporal coupling).
                        Default: joint.
  --output-dir PATH     Directory for saved results (default: output/).

Output
------
  output/real_m_est.npz   — NumPy archive with keys:
      m_est               : shape (n_epochs, n_stations, n_layers)  — mm per layer
      depth_weights       : shape (n_stations, n_layers)  — fractional contribution
                            of each depth layer to the TOTAL InSAR surface displacement
                            at that station (not the 0–300 m MLCW sum).
                            depth_weights[s].sum() == insar_coverage[s] ≤ 1.0.
                            The remainder (1 - coverage) originates below 300 m.
                            Stage-2 input.
      insar_coverage      : shape (n_stations,)  — fraction of total InSAR surface
                            displacement at each station that is explained by the
                            0–300 m MLCW zone. Values ≤ 1.0; residual represents
                            deep compaction below 300 m that cannot be resolved.
      cumulative_compaction: shape (n_stations, n_layers)  — total mm per layer,
                            summed across all epochs.
      station_names       : list of station names
      valid_depths_m      : list of MLCW depth levels in metres
      month_labels        : list of 'Month_N' strings loaded
      grid_rows           : int  output grid Y dimension
      grid_cols           : int  output grid X dimension
      full_grid_pixel_indices : station pixel indices in the 153×117 output grid

Notes
-----
- m values are in mm (monthly incremental layer compaction).
- m_est[t, s, l] = estimated compaction of layer l at station s during epoch t.
- Negative values can occur due to the NNLS bound being 0; the solver enforces
  m >= 0 so the result is physically non-negative.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.loader import build_real_dataset
from src.solvers_temporal import solve_independent_epochs, solve_joint_spacetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-data CRFP joint inversion.")
    p.add_argument("--data-dir", default="my_input_data/", type=Path)
    p.add_argument("--month-start", default=0, type=int)
    p.add_argument("--month-end", default=None, type=int)
    p.add_argument("--lam", default=1e-2, type=float)
    p.add_argument("--lam-t", default=1.0, type=float)
    p.add_argument("--sigma-insar", default=3.0, type=float)
    p.add_argument("--sigma-well", default=1.0, type=float)
    p.add_argument("--solver", choices=["joint", "independent"], default="joint")
    p.add_argument("--output-dir", default="output/", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real data ────────────────────────────────────────────────────
    print("Loading real data from:", args.data_dir)
    dataset, config, meta = build_real_dataset(
        data_dir=args.data_dir,
        month_start=args.month_start,
        month_end=args.month_end,
        lam=args.lam,
        lam_t=args.lam_t,
        sigma_insar=args.sigma_insar,
        sigma_well=args.sigma_well,
    )

    n_stations = config.n_pixels   # = grid_rows * grid_cols = 31 for real data
    n_layers = config.n_layers
    n_epochs = config.n_epochs
    print(f"  Stations : {n_stations}")
    print(f"  Depth levels (m): {meta['valid_depths_m']}")
    print(f"  Epochs   : {n_epochs}  ({meta['month_labels'][0]} – {meta['month_labels'][-1]})")
    print(f"  Model size: {config.total_voxels} voxels × {n_epochs} epochs = {config.total_voxels * n_epochs:,} unknowns")

    # ── Run inversion ─────────────────────────────────────────────────────
    print(f"\nRunning {args.solver} inversion (lam={args.lam}, lam_t={args.lam_t}) …")
    t0 = time.perf_counter()

    if args.solver == "joint":
        m_flat = solve_joint_spacetime(dataset)
    else:
        m_flat = solve_independent_epochs(dataset)

    elapsed = time.perf_counter() - t0
    print(f"  Solver finished in {elapsed:.1f} s")

    # ── Reshape and summarise ─────────────────────────────────────────────
    # m_flat layout: [epoch0_m (total_voxels,), epoch1_m, ..., epochT-1_m]
    # Reshape to (n_epochs, n_stations, n_layers)
    m_est = m_flat.reshape(n_epochs, n_stations, n_layers)

    print(f"\nResult summary:")
    print(f"  m_est shape : {m_est.shape}  (epochs × stations × depth layers)")
    print(f"  m_est range : {m_est.min():.3f} – {m_est.max():.3f} mm")
    print(f"  m_est mean  : {m_est.mean():.3f} mm")

    # Per-station cumulative compaction (sum over epochs and layers)
    cumulative_per_station = m_est.sum(axis=(0, 2))  # shape (n_stations,)
    print("\nTop 5 stations by total cumulative compaction (mm):")
    order = np.argsort(cumulative_per_station)[::-1][:5]
    for rank, idx in enumerate(order, 1):
        print(f"  {rank}. {meta['station_names'][idx]:20s}  {cumulative_per_station[idx]:.1f} mm")

    # ── Depth-weighting profiles (Stage-2 output) ─────────────────────────
    # cumulative[s, l] = total compaction (mm) summed over all epochs
    cumulative = m_est.sum(axis=0)                           # (n_stations, n_layers)

    # InSAR cumulative surface displacement per station (sum over all 83 epochs).
    # This is the physically correct denominator: weights will sum to ≤ 1.0.
    insar_cumulative = dataset.d_insar.reshape(n_epochs, n_stations).sum(axis=0)  # (n_stations,)
    insar_cumulative_safe = np.where(insar_cumulative > 0, insar_cumulative, 1.0)

    # Coverage fraction: fraction of total InSAR signal explained by 0–300 m MLCW.
    # Residual (1 - coverage) originates below 300 m and cannot be recovered.
    # Note: Can be > 1.0 if MLCW-derived m_est values exceed InSAR due to measurement
    # noise or well calibration biases. Values < 1.0 indicate deep compaction contribution.
    insar_coverage = cumulative.sum(axis=1) / insar_cumulative_safe  # (n_stations,)

    # Depth weights: each layer's fraction of TOTAL InSAR surface displacement.
    # depth_weights[s, l] = how much of the total InSAR signal at station s is
    # attributed to layer l (within 0–300 m). Sum = insar_coverage[s].
    depth_weights = cumulative / insar_cumulative_safe[:, np.newaxis]  # (n_stations, n_layers)

    valid_depths_m = meta["valid_depths_m"]
    mean_weights = depth_weights.mean(axis=0)                # average across stations
    top3_idx = np.argsort(mean_weights)[::-1][:3]
    print("\nTop 3 depth layers by mean fractional weighting (across all stations):")
    for i in top3_idx:
        print(f"  {valid_depths_m[i]:>5} m : {mean_weights[i] * 100:.1f}% of total InSAR surface")

    mean_coverage = insar_coverage.mean()
    print(f"\nMean 0–300 m coverage fraction across stations: {mean_coverage * 100:.1f}%")
    print(f"  (Residual {(1 - mean_coverage) * 100:.1f}% of surface signal originates below 300 m)")

    # ── Save results ──────────────────────────────────────────────────────
    out_path = args.output_dir / "real_m_est.npz"
    np.savez(
        out_path,
        m_est=m_est,
        depth_weights=depth_weights,
        cumulative_compaction=cumulative,
        insar_coverage=insar_coverage,
        station_names=np.array(meta["station_names"]),
        valid_depths_m=np.array(meta["valid_depths_m"]),
        month_labels=np.array(meta["month_labels"]),
        grid_rows=np.array(meta["grid_rows"]),
        grid_cols=np.array(meta["grid_cols"]),
        full_grid_pixel_indices=np.array(meta["full_grid_pixel_indices"]),
        x_twd97=np.array(meta["x_twd97"]),
        y_twd97=np.array(meta["y_twd97"]),
    )
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
