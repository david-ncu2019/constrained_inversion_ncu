import argparse
import logging
import time
from pathlib import Path

import numpy as np

from src.config import SystemConfig
from src.solvers_temporal import solve_independent_epochs, solve_joint_spacetime
from src.synthetic_temporal import generate_timeseries_world
from src.visualize_temporal import plot_timeseries_comparison

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Temporal Joint Inversion Solver")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--n-epochs", type=int, default=12, help="Number of time steps")
    parser.add_argument("--lam", type=float, default=1e-2, help="Spatial Tikhonov lambda")
    parser.add_argument("--lam-t", type=float, default=5.0, help="Temporal Tikhonov lambda")
    parser.add_argument("--sigma-insar", type=float, default=0.5, help="InSAR noise std dev")
    parser.add_argument("--sigma-well", type=float, default=0.1, help="Well noise std dev")
    args = parser.parse_args()

    # 1. Configuration
    config = SystemConfig(
        n_epochs=args.n_epochs,
        lam=args.lam,
        lam_t=args.lam_t,
        sigma_insar=args.sigma_insar,
        sigma_well=args.sigma_well,
    )

    logger.info(f"Initialized Temporal SystemConfig (Epochs: {config.n_epochs})")

    # 2. Generate Synthetic Time-Series Data
    logger.info("Generating synthetic time-series data...")
    t0 = time.time()
    dataset = generate_timeseries_world(config, seed=args.seed)
    t1 = time.time()
    logger.info(f"Data generation complete in {t1 - t0:.2f} seconds.")

    # 3. Independent Per-Epoch Inversion
    logger.info("--- Running Independent Per-Epoch Solver ---")
    t2 = time.time()
    m_indep = solve_independent_epochs(dataset)
    t3 = time.time()
    logger.info(f"Independent Solver complete in {t3 - t2:.2f} seconds.")

    # 4. Joint Space-Time Inversion
    logger.info("--- Running Joint Space-Time Solver ---")
    t4 = time.time()
    m_joint = solve_joint_spacetime(dataset)
    t5 = time.time()
    logger.info(f"Joint Solver complete in {t5 - t4:.2f} seconds.")

    # 5. Results
    mse_indep = np.mean((m_indep - dataset.m_true) ** 2)
    mse_joint = np.mean((m_joint - dataset.m_true) ** 2)

    print("\n" + "=" * 50)
    print(f"{'Method':<30} | {'MSE (mm^2)'}")
    print("-" * 50)
    print(f"{'Independent Per-Epoch':<30} | {mse_indep:.4f}")
    print(f"{'Joint Space-Time':<30} | {mse_joint:.4f}")
    print("=" * 50 + "\n")

    # 6. Visualizations
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    logger.info("Generating temporal profile plot...")

    # Pick a random voxel to inspect
    voxel_to_plot = 125

    plot_timeseries_comparison(
        config,
        dataset.m_true,
        m_indep,
        m_joint,
        voxel_idx=voxel_to_plot,
        save_path=str(out_dir / "temporal_profile_comparison.png"),
    )

    logger.info(f"Done. Plot saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
