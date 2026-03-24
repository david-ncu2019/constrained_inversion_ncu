import argparse
import logging
from pathlib import Path

import numpy as np

from src.config import SystemConfig
from src.solvers import solve_bayesian, solve_tikhonov
from src.synthetic import generate_synthetic_world
from src.visualize import plot_layer_comparison

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Constrained Inversion Toy Problem")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--lam", type=float, default=1e-2, help="Tikhonov lambda")
    parser.add_argument("--ell", type=float, default=3.0, help="Correlation length")
    parser.add_argument("--sigma-insar", type=float, default=0.5, help="InSAR noise std dev")
    parser.add_argument("--sigma-well", type=float, default=0.1, help="Well noise std dev")
    args = parser.parse_args()

    # 1. Configuration
    config = SystemConfig(
        lam=args.lam, ell=args.ell, sigma_insar=args.sigma_insar, sigma_well=args.sigma_well
    )

    logger.info(f"Initialized SystemConfig: {config}")

    # 2. Generate Synthetic Data
    logger.info(f"Generating synthetic data with seed {args.seed}...")
    dataset = generate_synthetic_world(config, seed=args.seed)

    logger.info(
        f"Data Shapes: m_true={dataset.m_true.shape}, d_insar={dataset.d_insar.shape}, w={dataset.w.shape}"
    )

    # 3. Path A: Tikhonov Regularization
    logger.info("Solving Path A: Tikhonov Regularization...")
    m_hat_tikhonov = solve_tikhonov(dataset)
    mse_tikhonov = np.mean((m_hat_tikhonov - dataset.m_true) ** 2)
    logger.info(f"Tikhonov Mean Squared Error: {mse_tikhonov:.4f}")

    # 4. Path B: Bayesian Posterior Estimation
    logger.info("Solving Path B: Bayesian Posterior Estimation...")
    m_post_bayesian, C_post = solve_bayesian(dataset)
    mse_bayesian = np.mean((m_post_bayesian - dataset.m_true) ** 2)
    logger.info(f"Bayesian Mean Squared Error: {mse_bayesian:.4f}")

    # 5. Visualizations
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    logger.info("Generating plots...")
    plot_layer_comparison(
        config,
        dataset.m_true,
        m_hat_tikhonov,
        title=f"Path A: Tikhonov Inversion (MSE: {mse_tikhonov:.2f})",
        save_path=str(out_dir / "tikhonov_comparison.png"),
    )

    plot_layer_comparison(
        config,
        dataset.m_true,
        m_post_bayesian,
        title=f"Path B: Bayesian Inversion (MSE: {mse_bayesian:.2f})",
        save_path=str(out_dir / "bayesian_comparison.png"),
    )

    logger.info(f"Done. Plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
