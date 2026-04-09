from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear

from src.config import SyntheticDataset
from src.system import build_G_matrix, build_I_wells, build_L_matrix, build_L_spatial
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


def solve_joint_spacetime_cvxpy(
    dataset: SyntheticDataset,
    x_twd97: list[float] | np.ndarray,
    y_twd97: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Joint space-time inversion using CVXPY + OSQP with a hard inequality
    constraint enforcing the physical deep-residual bound.

    Minimises::

        (1/σ_insar²) ‖G_st m − d_insar‖² + (1/σ_well²) ‖I_wells_st m − w‖²
        + λ_s ‖L_spatial_st m‖² + λ_t ‖L_t m‖²

    subject to::

        m ≥ 0                                  (non-negativity)
        C_ineq m ≤ d_insar_flat                (layer-sum ≤ cumulative InSAR)

    The inequality constraint prevents the 0–300 m layers from absorbing
    deep compaction signal.  It is reliable only when the dataset is in
    cumulative displacement space (``config.cumulative_space=True``).

    ``C_ineq`` has shape ``(n_stations * n_epochs, total_voxels * n_epochs)``.
    Row ``t * n_stations + s`` sums all ``n_layers`` model entries for station
    ``s`` at epoch ``t`` — identical to the InSAR rows of the forward operator.

    Parameters
    ----------
    dataset : SyntheticDataset
        Must have been built with ``cumulate=True`` in ``build_real_dataset``.
    x_twd97 : array-like, shape (n_stations,)
        Easting coordinates in TWD97 metres.
    y_twd97 : array-like, shape (n_stations,)
        Northing coordinates in TWD97 metres.

    Returns
    -------
    np.ndarray, shape (total_voxels * n_epochs,)
        Epoch-major model vector in cumulative compaction space.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "cvxpy is required for solve_joint_spacetime_cvxpy. "
            "Install it with: uv add cvxpy"
        ) from exc

    config = dataset.config
    T = config.n_epochs
    M = config.total_voxels
    n_stations = config.n_pixels
    n_layers = config.n_layers

    x_arr = np.asarray(x_twd97, dtype=np.float64)
    y_arr = np.asarray(y_twd97, dtype=np.float64)

    # ── Build single-epoch operators ─────────────────────────────────────
    G = build_G_matrix(config)
    I_w = build_I_wells(config)

    # Geographic spatial smoother (falls back to sequential if no threshold)
    if config.spatial_dist_threshold_m > 0:
        L_s = build_L_spatial(config, x_arr, y_arr)
    else:
        L_s = build_L_matrix(config)

    A_single = sp.vstack([G, I_w], format="csr")

    # ── Block-diagonalise across epochs ──────────────────────────────────
    A_st = build_block_diagonal_operator(A_single, T)   # (T*(n_insar+n_well), T*M)
    L_st = build_block_diagonal_operator(L_s, T)         # (T*n_edges*n_layers, T*M)
    L_t = build_Lt_matrix(config)                        # (M*(T-1), M*T)

    # ── Observation vectors ───────────────────────────────────────────────
    b_st = np.concatenate([
        dataset.d_insar.reshape(T, -1),
        dataset.w.reshape(T, -1),
    ], axis=1).flatten()

    # ── Inequality constraint: Σ_l m_cum[t,s,l] ≤ InSAR_cum[t,s] ────────
    # C_ineq has same sparsity pattern as the InSAR rows of A_st.
    # Reuse block_diag of G for the constraint matrix.
    C_ineq = build_block_diagonal_operator(G, T)          # (T*n_stations, T*M)
    d_ineq = dataset.d_insar                               # (T*n_stations,)

    # ── CVXPY problem ─────────────────────────────────────────────────────
    sigma_insar = config.sigma_insar
    sigma_well = config.sigma_well
    lam_s = config.lam
    lam_t = config.lam_t

    m = cp.Variable(T * M, nonneg=True)

    # Wrap sparse matrices as CVXPY sparse parameters for efficiency
    A_st_cp = cp.Constant(A_st)
    L_st_cp = cp.Constant(L_st)
    L_t_cp = cp.Constant(L_t)
    C_ineq_cp = cp.Constant(C_ineq)

    residual_data = A_st_cp @ m - b_st
    residual_spatial = L_st_cp @ m
    residual_temporal = L_t_cp @ m

    # Weight data terms by noise variances
    n_insar_rows = T * n_stations
    n_well_rows = T * config.n_well_obs
    weights_data = np.concatenate([
        np.full(n_insar_rows, 1.0 / sigma_insar**2),
        np.full(n_well_rows, 1.0 / sigma_well**2),
    ])

    objective = cp.Minimize(
        cp.sum(cp.multiply(weights_data, cp.square(residual_data)))
        + lam_s * cp.sum_squares(residual_spatial)
        + lam_t * cp.sum_squares(residual_temporal)
    )

    constraints = [C_ineq_cp @ m <= d_ineq]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.OSQP,
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=10000,
    )

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"CVXPY/OSQP solver did not converge. Status: {problem.status}"
        )

    m_hat: np.ndarray = np.maximum(m.value, 0.0)  # clip numerical negatives
    return m_hat
