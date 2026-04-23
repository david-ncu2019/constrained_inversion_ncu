"""
tune_hyperparams_optuna.py — Hyperparameter tuning via Optuna Bayesian optimization.

Optimizes lam and lam_t using fold 3 temporal CV.
  Train:    months 0–71  (Jan 2015 – Dec 2020)
  Validate: months 72–82 (Jan–Nov 2021)

Usage
-----
  uv run python tune_hyperparams_optuna.py [options]

Options
-------
  --n-trials INT          Number of Optuna trials (default: 30)
  --data-dir PATH         Input data directory (default: my_input_data/)
  --output-dir PATH       Output directory (default: output/tune_hyperparams_optuna/)
"""

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from typing import Optional, List, Dict, Any, Tuple

from tune_hyperparams import evaluate_one_config, _get_fold3

def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning via Optuna.")
    parser.add_argument("--n-trials", default=30, type=int)
    parser.add_argument("--data-dir", default="my_input_data/", type=Path)
    parser.add_argument("--output-dir", default="output/tune_hyperparams_optuna/", type=Path)
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument("--sigma-insar", default=3.0, type=float)
    parser.add_argument("--sigma-well", default=1.0, type=float)
    parser.add_argument("--prev-best", default=None, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold3_def = _get_fold3(args.data_dir)

    def objective(trial: optuna.Trial) -> float:
        # Discretize the search space in log10 scale to reduce granularity
        # For lam: 10^-5 to 10^0 in steps of 0.5 (e.g., 1e-5, 3.16e-5, 1e-4...)
        log_lam = trial.suggest_float("log_lam", -5.0, 0.0, step=0.5)
        lam = round(10 ** log_lam, 6)

        # For lam_t: 10^-2 to 10^1 in steps of 0.25 (e.g., 0.01, 0.017, 0.031...)
        log_lam_t = trial.suggest_float("log_lam_t", -2.0, 1.0, step=0.25)
        lam_t = round(10 ** log_lam_t, 4)

        # Evaluate
        result = evaluate_one_config(
            data_dir=args.data_dir,
            lam=lam,
            lam_t=lam_t,
            sigma_insar=args.sigma_insar,
            sigma_well=args.sigma_well,
            fold3_def=fold3_def,
            config_path=args.config,
        )
        
        # Log trial results to a CSV for tracking
        trial_results = {
            "trial": trial.number,
            "lam": lam,
            "lam_t": lam_t,
            "mean_rmse": result["mean_rmse_mm"],
            "elapsed_s": result["elapsed_s"]
        }
        pd.DataFrame([trial_results]).to_csv(
            args.output_dir / "optuna_trials.csv", 
            mode='a', 
            header=not (args.output_dir / "optuna_trials.csv").exists(), 
            index=False
        )

        return result["mean_rmse_mm"]

    # Use TPESampler with a specified number of startup trials for better efficiency
    sampler = optuna.samplers.TPESampler(n_startup_trials=5)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    if args.prev_best and args.prev_best.exists():
        print(f"Reading previous best parameters from {args.prev_best}...")
        prev_params = {}
        with open(args.prev_best, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=")
                    prev_params[k] = float(v)
        
        # Enqueue the specific values provided, converting back to log-scale
        # as expected by the discretized search space
        enqueue_dict = {}
        if "lam" in prev_params:
            enqueue_dict["log_lam"] = np.log10(prev_params["lam"])
        elif "log_lam" in prev_params:
            enqueue_dict["log_lam"] = prev_params["log_lam"]

        if "lam_t" in prev_params:
            enqueue_dict["log_lam_t"] = np.log10(prev_params["lam_t"])
        elif "log_lam_t" in prev_params:
            enqueue_dict["log_lam_t"] = prev_params["log_lam_t"]

        if enqueue_dict:
            print(f"Enqueueing trial: {enqueue_dict}")
            study.enqueue_trial(enqueue_dict)
    else:
        # Enqueue a good initial guess (in log10 space) to speed up convergence
        # log_lam = -2.0 corresponds to lam = 0.01
        # log_lam_t = 0.0 corresponds to lam_t = 1.0
        study.enqueue_trial({
            "log_lam": -2.0,
            "log_lam_t": 0.0
        })

    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Optuna Optimization Summary ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"  Value (Mean RMSE): {study.best_value:.4f} mm")
    print(f"  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save best params to a file
    best_lam = round(10 ** study.best_params["log_lam"], 6)
    best_lam_t = round(10 ** study.best_params["log_lam_t"], 4)

    best_params_path = args.output_dir / "best_params_optuna.txt"
    with open(best_params_path, "w") as f:
        f.write(f"lam={best_lam}\n")
        f.write(f"lam_t={best_lam_t}\n")
    
    print(f"\nBest parameters saved to: {best_params_path}")

if __name__ == "__main__":
    main()
