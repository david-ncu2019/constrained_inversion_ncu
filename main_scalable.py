import argparse
import logging
import time
from pathlib import Path

import numpy as np

from src.config import SystemConfig
from src.scalable_solvers import solve_bayesian_scalable
from src.synthetic import generate_synthetic_world
from src.visualize import plot_layer_comparison

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Scalable Constrained Inversion Solver")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--grid-size", type=int, default=40, help="Size of the horizontal grid (N x N)")
    parser.add_argument("--n-layers", type=int, default=5, help="Number of vertical layers")
    parser.add_argument("--num-wells", type=int, default=10, help="Number of sparse observation wells")
    parser.add_argument("--ell", type=float, default=5.0, help="Horizontal correlation length")
    parser.add_argument("--ell-z", type=float, default=0.5, help="Vertical correlation length")
    parser.add_argument("--sigma-insar", type=float, default=1.0, help="InSAR noise std dev")
    parser.add_argument("--sigma-well", type=float, default=0.2, help="Well noise std dev")
    args = parser.parse_args()

    # 1. Configuration (Scalable)
    config = SystemConfig.create_scalable(
        grid_size=args.grid_size,
        n_layers=args.n_layers,
        num_wells=args.num_wells,
        seed=args.seed,
        ell=args.ell,
        ell_z=args.ell_z,
        sigma_insar=args.sigma_insar,
        sigma_well=args.sigma_well,
    )

    logger.info("Initialized Scalable SystemConfig:")
    logger.info(f"  Grid Size: {config.grid_size}x{config.grid_size}")
    logger.info(f"  Layers: {config.n_layers}")
    logger.info(f"  Total Voxels (M): {config.total_voxels}")
    logger.info(f"  Number of Wells: {args.num_wells} (covering {config.n_well_obs} voxels)")
    logger.info(f"  Total Observations: {config.n_insar_obs + config.n_well_obs}")

    # 2. Generate Synthetic Data
    logger.info("Generating synthetic data (this may take a moment for large grids)...")
    t0 = time.time()
    dataset = generate_synthetic_world(config, seed=args.seed)
    t1 = time.time()
    logger.info(f"Data generation complete in {t1 - t0:.2f} seconds.")

    # 3. Evaluate Scalable Solver
    logger.info("--- Running Scalable Bayesian Solver ---")
    t2 = time.time()
    m_post, var_post = solve_bayesian_scalable(dataset, chunk_size=2000)
    t3 = time.time()
    logger.info(f"Solver complete in {t3 - t2:.2f} seconds.")

    # 4. Results
    mse = np.mean((m_post - dataset.m_true) ** 2)
    print("\n" + "=" * 50)
    print(f"{'Metric':<35} | {'Value'}")
    print("-" * 50)
    print(f"{'Mean Squared Error (MSE)':<35} | {mse:.4f} mm^2")
    print(f"{'Max Uncertainty (Variance)':<35} | {np.max(var_post):.4f} mm^2")
    print(f"{'Min Uncertainty (Variance)':<35} | {np.min(var_post):.4f} mm^2")
    print(f"{'Execution Time':<35} | {t3 - t2:.2f} s")
    print("=" * 50 + "\n")

    # 5. Visualizations
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    logger.info("Generating plots...")
    plot_layer_comparison(
        config,
        dataset.m_true,
        m_post,
        title=f"Scalable Bayesian Inversion (Grid {config.grid_size}x{config.grid_size}, Wells {args.num_wells}, MSE: {mse:.2f})",
        save_path=str(out_dir / "scalable_bayesian_comparison.png"),
    )

    logger.info(f"Done. Plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
