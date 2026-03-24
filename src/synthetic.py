import numpy as np
from scipy.ndimage import gaussian_filter

from src.config import SyntheticDataset, SystemConfig
from src.system import build_G_matrix, build_I_wells


def generate_synthetic_world(config: SystemConfig, seed: int = 42) -> SyntheticDataset:
    """
    Generates a synthetic true underground model and noisy observations.

    1. Creates a 3D spatially correlated true model (m_true).
    2. Simulates satellite (InSAR) and well (MLCW) observations.
    3. Adds Gaussian noise to the observations.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate m_true
    # We want a spatially correlated field. The easiest way is to generate
    # white noise in the 3D grid and apply a Gaussian smoothing filter.
    grid_shape = (config.grid_size, config.grid_size, config.n_layers)
    white_noise_grid = rng.standard_normal(grid_shape)

    # Apply Gaussian filter to create spatial correlation
    # We use different sigma for horizontal vs vertical to mimic geology
    smoothed_grid = gaussian_filter(white_noise_grid, sigma=(2.0, 2.0, 0.5))

    # Scale and shift to make it look like realistic compaction (e.g., all positive)
    # Let's say mean compaction per layer is around 5 mm, with some variation
    m_true_grid = 5.0 + 3.0 * (smoothed_grid / np.std(smoothed_grid))
    m_true_grid = np.clip(m_true_grid, 0.0, None)  # Compaction should be positive

    # Flatten column-major to match the system formulation
    # The proposal says: "column-major order (all five layers of pixel 1, then pixel 2...)"
    # We need to be careful with reshaping.
    # Our grid is indexed by (row, col, layer).
    # Flat index i = 10*(row) + col for pixels.
    # We want m to be [pixel1_L1, pixel1_L2, ..., pixel100_L5]
    # Reshaping with order='C' or taking care of axes:
    # A standard reshape with order='C' over a flattened (pixel, layer) array is what we want.
    m_true_grid_2d = m_true_grid.reshape((config.n_pixels, config.n_layers))
    m_true = m_true_grid_2d.flatten()  # Flattens to [p0_l0, p0_l1..., p1_l0...]

    # 2. Simulate noise-free observations
    G = build_G_matrix(config)
    I_w = build_I_wells(config)

    d_insar_perfect = G @ m_true
    w_perfect = I_w @ m_true

    # 3. Add noise
    noise_insar = rng.normal(0, config.sigma_insar, size=config.n_insar_obs)
    noise_well = rng.normal(0, config.sigma_well, size=config.n_well_obs)

    d_insar = d_insar_perfect + noise_insar
    w = w_perfect + noise_well

    return SyntheticDataset(config=config, m_true=m_true, d_insar=d_insar, w=w)
