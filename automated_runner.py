import os
import subprocess
import time
from pathlib import Path
import numpy as np

# Need to import build_real_dataset to get the "true" well observations for MSE
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.loader import build_real_dataset

def main():
    base_dir = Path(r"D:\110_PROJECT_002\constrained_inversion_scripts")
    test_run_dir = base_dir / "test_run"
    test_run_dir.mkdir(exist_ok=True)
    
    # Discover the next run number
    existing_runs = [d.name for d in test_run_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_numbers = [int(r.split("_")[1]) for r in existing_runs if r.split("_")[1].isdigit()]
    next_run = max(run_numbers) + 1 if run_numbers else 1

    # Grid search parameters
    lam_t_values = [1.0, 3.0, 4.0, 5.0, 6.0]
    solvers = ["independent", "joint"] # independent first as it's faster, joint is recommended
    
    # Summary report file
    summary_file = test_run_dir / "summary_grid_search.md"
    if not summary_file.exists():
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("# Grid Search Summary\n\n")
            f.write("| Run | Solver | lam_t | MSE (Well) | Time (s) |\n")
            f.write("|---|---|---|---|---|\n")

    # Load dataset ONCE to get true well measurements (dataset.w) for MSE calculation
    print("Loading real dataset to obtain well observation ground truth for MSE calculation...")
    dataset, config, meta = build_real_dataset(data_dir=base_dir / "my_input_data")
    w_true = dataset.w  # shape: (n_epochs * n_stations * n_layers,)

    for solver in solvers:
        for lam_t in lam_t_values:
            run_name = f"run_{next_run}"
            run_dir = test_run_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n======================================")
            print(f"Starting {run_name}: Solver = {solver}, lam_t = {lam_t}")
            print(f"======================================")
            
            # Save config
            config_text = f"solver: {solver}\nlam_t: {lam_t}\nlam: 1e-2\n"
            (run_dir / "config.txt").write_text(config_text, encoding="utf-8")
            
            # Command to run main_real.py
            cmd = [
                "uv", "run", "python", "main_real.py",
                "--lam-t", str(lam_t),
                "--solver", solver,
                "--output-dir", str(run_dir)
            ]
            
            start_time = time.time()
            try:
                # Run the subprocess
                subprocess.run(cmd, cwd=str(base_dir), check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running {run_name}. Skipping to next.")
                continue
                
            elapsed_time = time.time() - start_time
            
            # Calculate MSE
            npz_path = run_dir / "real_m_est.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                m_est = data["m_est"] # shape (n_epochs, n_stations, n_layers)
                m_est_flat = m_est.flatten()
                
                # Check for physical plausibility (all >= 0)
                min_m = float(np.min(m_est_flat))
                
                mse = float(np.mean((m_est_flat - w_true) ** 2))
                
                # Generate report
                report_text = f"# Report for {run_name}\n\n"
                report_text += f"**Solver:** {solver}\n"
                report_text += f"**Temporal Regularization (lam_t):** {lam_t}\n\n"
                report_text += f"## Metrics\n"
                report_text += f"- **MSE at MLCW observation points:** {mse:.4f} mm^2\n"
                report_text += f"- **Minimum compaction value:** {min_m:.4f} mm (Should be >= 0 for physical plausibility)\n"
                report_text += f"- **Execution time:** {elapsed_time:.1f} seconds\n"
                
                (run_dir / "report.md").write_text(report_text, encoding="utf-8")
                
                # Append to summary
                with open(summary_file, "a", encoding="utf-8") as f:
                    f.write(f"| {run_name} | {solver} | {lam_t} | {mse:.4f} | {elapsed_time:.1f} |\n")
                    
                print(f"--> Finished {run_name}. MSE: {mse:.4f}")
            else:
                print(f"--> Failed to find output NPZ for {run_name}")

            next_run += 1

if __name__ == "__main__":
    main()
