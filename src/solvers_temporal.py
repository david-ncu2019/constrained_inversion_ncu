import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear

from src.config import SyntheticDataset
from src.system import build_G_matrix, build_I_wells, build_L_matrix
from src.system_temporal import build_block_diagonal_operator, build_Lt_matrix


def solve_independent_epochs(dataset: SyntheticDataset) -> np.ndarray:
    """
    Solves each time epoch completely independently using Tikhonov NNLS.
    """
    config = dataset.config
    M = config.total_voxels
    T = config.n_epochs

    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    L = build_L_matrix(config)

    A_single = sp.vstack([G, I_w], format="csr")
    sqrt_lam = np.sqrt(config.lam)
    A_aug = sp.vstack([A_single, sqrt_lam * L], format="csr")

    m_est_all = np.zeros(M * T, dtype=np.float64)

    # Reshape observations to (T, N_obs)
    d_insar_t = dataset.d_insar.reshape(T, -1)
    w_t = dataset.w.reshape(T, -1)

    for t in range(T):
        b_single = np.concatenate([d_insar_t[t], w_t[t]])
        b_aug = np.concatenate([b_single, np.zeros(L.shape[0])])

        result = lsq_linear(A_aug, b_aug, bounds=(0, np.inf))

        start_idx = t * M
        m_est_all[start_idx : start_idx + M] = result.x

    return m_est_all


def solve_joint_spacetime(dataset: SyntheticDataset) -> np.ndarray:
    """
    Solves all epochs simultaneously, enforcing both spatial and temporal smoothness.
    """
    config = dataset.config
    T = config.n_epochs

    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    L_s = build_L_matrix(config)

    A_single = sp.vstack([G, I_w], format="csr")

    # 1. Block diagonalize spatial operators
    A_st = build_block_diagonal_operator(A_single, T)
    L_st = build_block_diagonal_operator(L_s, T)

    # 2. Temporal operator
    L_t = build_Lt_matrix(config)

    # 3. Augment system
    sqrt_lam_s = np.sqrt(config.lam)
    sqrt_lam_t = np.sqrt(config.lam_t)

    A_aug = sp.vstack([
        A_st,
        sqrt_lam_s * L_st,
        sqrt_lam_t * L_t
    ], format="csr")

    # Stack all b
    b_st = np.concatenate([
        dataset.d_insar.reshape(T, -1),
        dataset.w.reshape(T, -1)
    ], axis=1).flatten()

    # Pad b with zeros for both spatial and temporal regularization
    b_aug = np.concatenate([
        b_st,
        np.zeros(L_st.shape[0]),
        np.zeros(L_t.shape[0])
    ])

    # 4. Solve jointly with non-negativity constraint
    # lsq_linear handles sparse matrices natively
    result = lsq_linear(A_aug, b_aug, bounds=(0, np.inf))

    m_hat: np.ndarray = result.x
    return m_hat
