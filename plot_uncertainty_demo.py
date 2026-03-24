import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.config import SystemConfig
from src.synthetic import generate_synthetic_world
from src.solvers import solve_bayesian_anisotropic

def main():
    base_dir = Path(r"D:\110_PROJECT_002\constrained_inversion_scripts")
    out_dir = base_dir / "test_run" / "visualizations"
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Initializing synthetic grid for uncertainty demonstration...")
    # 10x10 grid, 5 layers. 5 wells distributed around.
    config = SystemConfig.create_default(grid_size=10, n_layers=5, num_wells=5, seed=42)
    
    dataset = generate_synthetic_world(config, seed=42)

    print("Running Bayesian Anisotropic Solver...")
    m_post, C_post = solve_bayesian_anisotropic(dataset)

    # C_post is the posterior covariance matrix (total_voxels x total_voxels).
    # The diagonal elements represent the variance (uncertainty) of each voxel.
    variance = np.diag(C_post)
    std_dev = np.sqrt(variance) # Standard deviation

    # Reshape to 3D grid
    std_dev_3d = std_dev.reshape((config.grid_size, config.grid_size, config.n_layers))
    
    # Let's plot the uncertainty for the first depth layer (layer 0)
    layer_0_std = std_dev_3d[:, :, 0]

    # Find the X, Y coordinates of the 5 wells to overlay on the plot
    well_coords = []
    for w_idx in config.well_indices:
        r = w_idx // config.grid_size
        c = w_idx % config.grid_size
        well_coords.append((c, r)) # (x, y)

    print("Generating Uncertainty Heatmap...")
    plt.figure(figsize=(8, 6))
    
    # We use 'magma_r' or 'viridis' so higher uncertainty is brighter/darker
    im = plt.imshow(layer_0_std, cmap='viridis', origin='upper')
    plt.colorbar(im, label='Posterior Standard Deviation (Uncertainty in mm)')
    
    # Plot wells
    for idx, (wx, wy) in enumerate(well_coords):
        if idx == 0:
            plt.scatter(wx, wy, c='red', marker='X', s=100, label='MLCW Well Locations')
        else:
            plt.scatter(wx, wy, c='red', marker='X', s=100)
            
    plt.title('Uncertainty Map (Bayesian Inversion)\nNotice lower uncertainty near wells', fontsize=14)
    plt.xlabel('Grid X', fontsize=12)
    plt.ylabel('Grid Y', fontsize=12)
    plt.legend()
    
    save_path = out_dir / 'bayesian_uncertainty_map.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Quantitative evaluation complete. Min Uncertainty: {std_dev.min():.4f}, Max Uncertainty: {std_dev.max():.4f}")
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    main()
