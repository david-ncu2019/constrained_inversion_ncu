import argparse
import logging

import numpy as np

from src.config import SystemConfig
from src.solvers import (
    solve_bayesian,
    solve_bayesian_anisotropic,
    solve_bayesian_depth_prior,
    solve_tikhonov,
    solve_tikhonov_nnls,
)
from src.synthetic import generate_synthetic_world

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Enhanced Constrained Inversion Solvers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--lam", type=float, default=1e-2, help="Tikhonov lambda")
    parser.add_argument("--ell", type=float, default=3.0, help="Horizontal correlation length")
    parser.add_argument("--ell-z", type=float, default=0.5, help="Vertical correlation length")
    parser.add_argument("--sigma-insar", type=float, default=0.5, help="InSAR noise std dev")
    parser.add_argument("--sigma-well", type=float, default=0.1, help="Well noise std dev")
    args = parser.parse_args()

    # 1. Configuration
    config = SystemConfig(
        lam=args.lam,
        ell=args.ell,
        ell_z=args.ell_z,
        sigma_insar=args.sigma_insar,
        sigma_well=args.sigma_well,
    )

    logger.info(f"Initialized SystemConfig: {config}")

    # 2. Generate Synthetic Data
    logger.info(f"Generating synthetic data with seed {args.seed}...")
    dataset = generate_synthetic_world(config, seed=args.seed)

    logger.info(
        f"Data Shapes: m_true={dataset.m_true.shape}, d_insar={dataset.d_insar.shape}, w={dataset.w.shape}"
    )

    # 3. Evaluate Solvers
    results = {}

    logger.info("--- Running Solvers ---")

    # Baseline Tikhonov
    m_tikhonov = solve_tikhonov(dataset)
    results["Tikhonov (Baseline)"] = np.mean((m_tikhonov - dataset.m_true) ** 2)

    # Enhanced Tikhonov NNLS
    m_tikhonov_nnls = solve_tikhonov_nnls(dataset)
    results["Tikhonov (Non-Negative)"] = np.mean((m_tikhonov_nnls - dataset.m_true) ** 2)

    # Baseline Bayesian
    m_bayesian, _ = solve_bayesian(dataset)
    results["Bayesian (Baseline)"] = np.mean((m_bayesian - dataset.m_true) ** 2)

    # Enhanced Bayesian Anisotropic
    m_bayes_aniso, _ = solve_bayesian_anisotropic(dataset)
    results["Bayesian (Anisotropic)"] = np.mean((m_bayes_aniso - dataset.m_true) ** 2)

    # Enhanced Bayesian Depth Prior
    m_bayes_depth, _ = solve_bayesian_depth_prior(dataset)
    results["Bayesian (Depth Prior + Aniso)"] = np.mean((m_bayes_depth - dataset.m_true) ** 2)

    # 4. Print Summary Table
    print("\n" + "=" * 50)
    print(f"{'Solver Variant':<35} | {'MSE (mm^2)':<10}")
    print("-" * 50)
    for name, mse in results.items():
        print(f"{name:<35} | {mse:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
