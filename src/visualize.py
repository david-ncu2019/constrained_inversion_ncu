from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.config import SystemConfig


def plot_layer_comparison(
    config: SystemConfig, m_true: np.ndarray, m_est: np.ndarray, title: str, save_path: Optional[str] = None
) -> None:
    """
    Plots a layer-by-layer comparison between the true model and estimated model.
    """
    m_true_3d = m_true.reshape((config.grid_size, config.grid_size, config.n_layers))
    m_est_3d = m_est.reshape((config.grid_size, config.grid_size, config.n_layers))

    fig, axes = plt.subplots(config.n_layers, 3, figsize=(12, 3 * config.n_layers))
    fig.suptitle(title, fontsize=16)

    # Calculate global vmin and vmax for consistent color scaling
    vmin = min(m_true.min(), m_est.min())
    vmax = max(m_true.max(), m_est.max())

    for layer in range(config.n_layers):
        true_layer = m_true_3d[:, :, layer]
        est_layer = m_est_3d[:, :, layer]
        diff_layer = est_layer - true_layer

        # True
        ax_true = axes[layer, 0]
        im1 = ax_true.imshow(true_layer, vmin=vmin, vmax=vmax, cmap="viridis")
        ax_true.set_title(f"True Layer {layer + 1}")
        fig.colorbar(im1, ax=ax_true)

        # Estimated
        ax_est = axes[layer, 1]
        im2 = ax_est.imshow(est_layer, vmin=vmin, vmax=vmax, cmap="viridis")
        ax_est.set_title(f"Estimated Layer {layer + 1}")
        fig.colorbar(im2, ax=ax_est)

        # Difference
        ax_diff = axes[layer, 2]
        # Diff map uses a divergent colormap centered at zero
        diff_max = np.abs(diff_layer).max() + 1e-6
        im3 = ax_diff.imshow(diff_layer, vmin=-diff_max, vmax=diff_max, cmap="RdBu_r")
        ax_diff.set_title("Difference (Est - True)")
        fig.colorbar(im3, ax=ax_diff)

        # Mark well locations on the True plot
        for well_idx in config.well_indices:
            row = well_idx // config.grid_size
            col = well_idx % config.grid_size
            ax_true.plot(
                col,
                row,
                "rX",
                markersize=10,
                label="Well" if well_idx == config.well_indices[0] else "",
            )
            if layer == 0 and well_idx == config.well_indices[-1]:
                ax_true.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)
