import numpy as np
from scipy.ndimage import gaussian_filter

from src.config import SyntheticDataset, SystemConfig
from src.system import build_G_matrix, build_I_wells


def generate_timeseries_world(config: SystemConfig, seed: int = 42) -> SyntheticDataset:
    """
    Generates a synthetic true underground model and noisy observations over T epochs.

    Returns:
        SyntheticDataset where arrays are sized (N_obs * T).
        The data is stacked epoch by epoch: [Epoch1_data, Epoch2_data, ..., EpochT_data]
    """
    rng = np.random.default_rng(seed)

    # 1. Base spatial field
    grid_shape = (config.grid_size, config.grid_size, config.n_layers)
    white_noise_grid = rng.standard_normal(grid_shape)
    smoothed_grid = gaussian_filter(white_noise_grid, sigma=(2.0, 2.0, 0.5))
    base_compaction = 1.0 + 1.0 * (smoothed_grid / np.std(smoothed_grid))
    base_compaction = np.clip(base_compaction, 0.0, None)  # Base compaction strictly positive
    base_m_true = base_compaction.reshape((config.n_pixels, config.n_layers)).flatten()

    M = config.total_voxels
    T = config.n_epochs

    # 2. Time evolution
    # We create a logistic/sigmoidal growth curve to mimic real consolidation
    t_steps = np.linspace(-3, 3, T)
    growth_curve = 1 / (1 + np.exp(-t_steps))

    # Normalize growth from 0.1 to 1.0
    growth_curve = 0.1 + 0.9 * (growth_curve - np.min(growth_curve)) / (
        np.max(growth_curve) - np.min(growth_curve)
    )

    m_true_all = np.zeros((T, M), dtype=np.float64)
    d_insar_all = np.zeros((T, config.n_insar_obs), dtype=np.float64)
    w_all = np.zeros((T, config.n_well_obs), dtype=np.float64)

    G = build_G_matrix(config)
    I_w = build_I_wells(config)

    for t in range(T):
        # The true compaction at time t is the base scaled by the growth curve
        # Plus a tiny bit of random drift (random walk)
        random_drift = rng.normal(0, 0.05, size=M)
        m_t = base_m_true * growth_curve[t] + random_drift
        m_t = np.clip(m_t, 0.0, None)

        # Ensure strict monotonically increasing (compaction only goes one way)
        if t > 0:
            m_t = np.maximum(m_t, m_true_all[t - 1, :])

        m_true_all[t, :] = m_t

        # Forward simulate observations
        d_insar_perfect = G @ m_t
        w_perfect = I_w @ m_t

        # Add noise
        noise_insar = rng.normal(0, config.sigma_insar, size=config.n_insar_obs)
        noise_well = rng.normal(0, config.sigma_well, size=config.n_well_obs)

        d_insar_all[t, :] = d_insar_perfect + noise_insar
        w_all[t, :] = w_perfect + noise_well

    # Flatten the results.
    # The order is: All of Epoch 1, then all of Epoch 2, etc.
    return SyntheticDataset(
        config=config, m_true=m_true_all.flatten(), d_insar=d_insar_all.flatten(), w=w_all.flatten()
    )
