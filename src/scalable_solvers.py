import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from src.config import SyntheticDataset
from src.system import build_G_matrix, build_I_wells


def solve_bayesian_scalable(
    dataset: SyntheticDataset, chunk_size: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scalable Bayesian Posterior Estimation with Anisotropic Covariance.
    Instead of forming the massive M x M prior covariance matrix C_m,
    it computes the necessary projection S = C_m * A.T in memory-efficient chunks.

    Returns:
        m_post: Posterior mean estimate (M,)
        var_post: Posterior variance (diagonal of C_post) (M,)
    """
    config = dataset.config
    M = config.total_voxels

    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    A = sp.vstack([G, I_w], format="csr")
    b = np.concatenate([dataset.d_insar, dataset.w])

    # 1. Depth-Dependent Prior Mean (m_0)
    w_reshaped = dataset.w.reshape(-1, config.n_layers)  # (n_wells, n_layers)
    layer_means = np.mean(w_reshaped, axis=0)  # shape (n_layers,)

    p_indices = np.arange(M)
    layer_indices = p_indices % config.n_layers
    m_0 = layer_means[layer_indices]

    # 2. Coordinates for Anisotropic distance
    pixel_indices = p_indices // config.n_layers
    rows = pixel_indices // config.grid_size
    cols = pixel_indices % config.grid_size

    scaled_rows = rows / config.ell
    scaled_cols = cols / config.ell
    scaled_layers = layer_indices / config.ell_z

    coords = np.column_stack((scaled_rows, scaled_cols, scaled_layers))

    # 3. Compute S = C_m @ A.T in chunks to save memory
    # A is (N_obs x M), A.T is (M x N_obs)
    # S will be (M x N_obs)
    N_obs = A.shape[0]
    S = np.zeros((M, N_obs), dtype=np.float64)
    A_T = A.T.toarray()  # M x N_obs (dense is fine since N_obs is relatively small)

    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        coords_chunk = coords[i:end_i]

        # cdist computes distances from chunk (size x 3) to all coords (M x 3)
        # distances_chunk is (size x M)
        distances_chunk = cdist(coords_chunk, coords, metric="euclidean")
        C_m_chunk = config.prior_variance * np.exp(-distances_chunk)

        # Multiply by A.T to project into observation space
        # (size x M) @ (M x N_obs) -> (size x N_obs)
        S[i:end_i, :] = C_m_chunk @ A_T

    # 4. Compute T = A @ C_m @ A.T + C_n  == A @ S + C_n
    var_insar = np.ones(config.n_insar_obs) * (config.sigma_insar**2)
    var_well = np.ones(config.n_well_obs) * (config.sigma_well**2)
    C_n = np.diag(np.concatenate([var_insar, var_well]))

    T = A @ S + C_n  # (N_obs x N_obs)

    # 5. Compute K = S @ T^{-1}
    T_inv = np.linalg.inv(T)
    K = S @ T_inv  # (M x N_obs)

    # 6. Posterior Mean
    residual = b - A @ m_0
    m_post = m_0 + K @ residual

    # 7. Posterior Variance (Diagonal of C_post)
    # diag(C_post) = diag(C_m) - diag(K @ A @ C_m) = diag(C_m) - diag(K @ S.T)
    # diag(K @ S.T) is the sum of element-wise multiplication of K and S
    variance_reduction = np.sum(K * S, axis=1)
    var_post = config.prior_variance - variance_reduction

    return m_post, var_post
