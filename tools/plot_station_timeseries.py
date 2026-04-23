import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import sys
import argparse

# Add src to python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.loader import build_real_dataset, parse_dataset_config

def main():
    parser = argparse.ArgumentParser(description="Plot predicted vs observed cumulative compaction timeseries.")
    parser.add_argument("--npz", default="output/real_m_est.npz", type=Path)
    parser.add_argument("--config", default="configs/pipeline_config.ini", type=Path)
    parser.add_argument("--data-dir", default="my_input_data", type=str)
    args = parser.parse_args()

    npz_path = project_root / args.npz
    if not npz_path.exists():
        print(f"Error: {npz_path} does not exist. Please run main_real.py first.")
        sys.exit(1)
        
    print(f"Loading predictions from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    m_est = data['m_est']  # shape: (n_epochs, n_stations, n_layers)
    station_names = data['station_names']
    valid_depths_m = data['valid_depths_m']
    month_labels = data['month_labels']
    
    n_epochs, n_stations, n_layers = m_est.shape
    
    print("Loading real dataset to get observations...")
    dataset_kwargs = parse_dataset_config(project_root / args.config)
    dataset, config, meta = build_real_dataset(
        data_dir=project_root / args.data_dir,
        cumulate=True,
        **dataset_kwargs
    )
    
    # In loader.py, the well observations `w` are populated such that:
    # d_insar_matrix is shape (n_stations, n_epochs)
    # w_matrix is shape (n_stations * n_layers, n_epochs)
    # w_flat is w_matrix.T.flatten() -> [epoch0 (n_well), epoch1, ...]
    # We want it back as (n_epochs, n_stations, n_layers)
    # w_flat.reshape(n_epochs, n_stations * n_layers) recovers w_matrix.T
    w_matrix_T = dataset.w.reshape(n_epochs, n_stations * n_layers)
    # Reshape each epoch's n_well back to (n_stations, n_layers)
    w_matrix_3d = w_matrix_T.reshape(n_epochs, n_stations, n_layers)
    
    out_dir = project_root / "output" / "timeseries_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Plotting for {n_stations} stations (up to {n_layers} layers each)...")
    
    x_vals = np.arange(n_epochs)
    
    for s_idx, station in enumerate(station_names):
        max_subplots = 30
        n_figs = math.ceil(n_layers / max_subplots)
        
        for fig_idx in range(n_figs):
            start_l = fig_idx * max_subplots
            end_l = min((fig_idx + 1) * max_subplots, n_layers)
            n_current_layers = end_l - start_l
            
            cols = min(5, n_current_layers)
            rows = math.ceil(n_current_layers / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
            fig.suptitle(f'Cumulative Compaction for Station {station} (Fig {fig_idx+1}/{n_figs})', fontsize=16)
            
            for i, l_idx in enumerate(range(start_l, end_l)):
                ax = axes[i // cols, i % cols]
                depth = valid_depths_m[l_idx]
                
                pred_series = m_est[:, s_idx, l_idx]
                obs_series = w_matrix_3d[:, s_idx, l_idx]
                
                # Check if observations exist for this layer.
                # If all are zero (or effectively zero), assume no valid observations.
                has_obs = not np.allclose(obs_series, 0.0, atol=1e-6)
                
                ax.plot(x_vals, pred_series, label='Predicted', color='blue', linewidth=2)
                
                if has_obs:
                    ax.plot(x_vals, obs_series, label='Observed', color='red', linestyle='--', alpha=0.7)
                    
                    # Calculate metrics
                    rmse = np.sqrt(np.mean((pred_series - obs_series)**2))
                    mae = np.mean(np.abs(pred_series - obs_series))
                    metrics_txt = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}"
                    
                    # Add text box inside the plot
                    ax.text(0.05, 0.95, metrics_txt, transform=ax.transAxes, 
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'Depth: {depth}m')
                ax.set_xlabel('Epoch (Months)')
                ax.set_ylabel('Compaction (mm)')
                ax.grid(True, linestyle=':', alpha=0.6)
                if i == 0:
                    ax.legend()
            
            # Hide empty subplots
            for j in range(n_current_layers, rows * cols):
                fig.delaxes(axes[j // cols, j % cols])
                
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            if n_figs > 1:
                filename = f"{station}_timeseries_{fig_idx+1}.png"
            else:
                filename = f"{station}_timeseries_1.png"
                
            save_path = out_dir / filename
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            
    print(f"All plots saved to {out_dir}")

if __name__ == '__main__':
    main()
