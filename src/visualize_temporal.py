import matplotlib.pyplot as plt
import numpy as np

from src.config import SystemConfig


def plot_timeseries_comparison(
    config: SystemConfig,
    dataset_true: np.ndarray,
    m_indep: np.ndarray,
    m_joint: np.ndarray,
    voxel_idx: int,
    save_path: str,
) -> None:
    """
    Plots the true vs estimated compaction over time for a single specific voxel.
    This clearly demonstrates the smoothing effect of the joint inversion.
    """
    T = config.n_epochs
    M = config.total_voxels

    # Extract the timeseries for the specific voxel
    t_steps = np.arange(T)

    # Reshape vectors back to (T, M) to easily slice out the voxel
    true_ts = dataset_true.reshape(T, M)[:, voxel_idx]
    indep_ts = m_indep.reshape(T, M)[:, voxel_idx]
    joint_ts = m_joint.reshape(T, M)[:, voxel_idx]

    plt.figure(figsize=(10, 6))

    plt.plot(t_steps, true_ts, "k--", linewidth=2, label="True Compaction")
    plt.plot(t_steps, indep_ts, "r-o", alpha=0.7, label="Independent Epoch Inversion")
    plt.plot(t_steps, joint_ts, "b-s", alpha=0.9, label="Joint Space-Time Inversion")

    plt.title(f"Temporal Compaction Profile for Voxel {voxel_idx}")
    plt.xlabel("Epoch (Time Step)")
    plt.ylabel("Compaction (mm)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
