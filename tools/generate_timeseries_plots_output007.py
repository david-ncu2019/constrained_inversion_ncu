import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import configparser
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.insert(0, 'D:/110_PROJECT_002/constrained_inversion_scripts')
from src.loader import build_real_dataset

def main():
    output_dir = Path('D:/110_PROJECT_002/constrained_inversion_scripts/my_input_data/output_007')
    plot_dir = output_dir / 'timeseries_plots'
    plot_dir.mkdir(exist_ok=True, parents=True)

    # Load config
    config = configparser.ConfigParser()
    config.read(output_dir / 'pipeline_config.ini')

    # Load real dataset
    print("Loading dataset...")
    dataset, sys_config, metadata = build_real_dataset(
        data_dir='D:/110_PROJECT_002/constrained_inversion_scripts/my_input_data',
        lam=0.01, lam_t=0.3,
        sigma_insar=3.0, sigma_well=1.0,
        month_start=int(config.get('Inversion', 'month_start', fallback=0)),
        month_end=None,
        cumulate=True,
        grid_metrics_file=config.get('Dataset', 'grid_metrics_file'),
        station_coords_file=config.get('Dataset', 'station_coords_file'),
        csv_dir_name=config.get('Dataset', 'csv_dir_name'),
        csv_name_pattern=config.get('Dataset', 'csv_name_pattern'),
        x_col=config.get('Dataset', 'x_col'),
        y_col=config.get('Dataset', 'y_col'),
        depth_col=config.get('Dataset', 'depth_col'),
        month_col_prefix=config.get('Dataset', 'month_col_prefix')
    )

    n_epochs = sys_config.n_epochs
    n_stations = len(metadata['station_names'])
    n_layers = sys_config.n_layers

    # w is [epoch0_data, epoch1_data...] where each epoch is (n_stations * n_layers)
    w_T = dataset.w.reshape((n_epochs, n_stations * n_layers))
    w_matrix = w_T.T # (n_stations * n_layers, n_epochs)
    w_cube = w_matrix.reshape((n_stations, n_layers, n_epochs))

    # Load m_est
    print("Loading m_est...")
    npz_data = np.load(output_dir / 'real_m_est.npz')
    m_est = npz_data['m_est'] # (n_epochs, n_stations, n_layers)
    m_est_cube = np.transpose(m_est, (1, 2, 0)) # (n_stations, n_layers, n_epochs)

    station_names = metadata['station_names']
    valid_depths = metadata['valid_depths_m']
    months = np.arange(n_epochs)

    # Plotting parameters
    plots_per_figure = 30
    rows = 6
    cols = 5

    print(f"Generating plots for {n_stations} stations...")
    for s_idx, station in enumerate(station_names):
        n_figs = int(np.ceil(n_layers / plots_per_figure))
        
        for fig_idx in range(n_figs):
            fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharex=True)
            axes = axes.flatten()
            
            fig.suptitle(f'Station: {station} - Cumulative Compaction Timeseries (Part {fig_idx+1})', fontsize=16)
            
            start_layer = fig_idx * plots_per_figure
            end_layer = min(start_layer + plots_per_figure, n_layers)
            
            for i, l_idx in enumerate(range(start_layer, end_layer)):
                ax = axes[i]
                
                obs = w_cube[s_idx, l_idx, :]
                pred = m_est_cube[s_idx, l_idx, :]
                
                ax.plot(months, pred, label='Predicted', color='blue', linewidth=2)
                
                # Check if there are valid observations
                valid_mask = ~np.isnan(obs)
                if np.any(valid_mask):
                    ax.plot(months[valid_mask], obs[valid_mask], 'ro-', label='Observed', markersize=4, linewidth=1)
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(obs[valid_mask], pred[valid_mask]))
                    mae = mean_absolute_error(obs[valid_mask], pred[valid_mask])
                    metrics_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}'
                    
                    # Add metrics text box
                    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                ax.set_title(f'Depth: {valid_depths[l_idx]}m')
                ax.grid(True, linestyle='--', alpha=0.6)
                if i == 0:
                    ax.legend(loc='upper right')
            
            # Hide unused subplots
            for i in range(end_layer - start_layer, len(axes)):
                axes[i].set_visible(False)
            
            # Set common labels
            fig.text(0.5, 0.04, 'Epoch (Month)', ha='center', fontsize=12)
            fig.text(0.04, 0.5, 'Cumulative Compaction (mm)', va='center', rotation='vertical', fontsize=12)
            
            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
            
            # Save figure
            if n_figs > 1:
                filename = f"{station}_timeseries_part{fig_idx+1}.png"
            else:
                filename = f"{station}_timeseries_1.png"
                
            save_path = plot_dir / filename
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            
    print(f"Successfully generated timeseries plots in {plot_dir}")

if __name__ == '__main__':
    main()
