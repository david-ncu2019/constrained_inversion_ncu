import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import lsqr
from scipy.spatial.distance import cdist

from src.config import SyntheticDataset
from src.system import build_G_matrix, build_I_wells, build_L_matrix


def solve_tikhonov(dataset: SyntheticDataset) -> np.ndarray:
    """
    Path A: Tikhonov Regularization
    Minimizes: ||A*m - b||^2 + lam * ||L*m||^2
    Solved via augmented least squares.
    """
    config = dataset.config

    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    L = build_L_matrix(config)

    # Stack the forward operator A and observations b
    A = sp.vstack([G, I_w], format="csr")
    b = np.concatenate([dataset.d_insar, dataset.w])

    # Augment the system with the regularization term
    # lam * ||L*m||^2 => || sqrt(lam) * L * m ||^2
    sqrt_lam = np.sqrt(config.lam)
    A_aug = sp.vstack([A, sqrt_lam * L], format="csr")

    # Pad b with zeros for the regularization equations
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])

    # Solve the augmented system using LSQR (suitable for sparse large scale)
    # lsqr returns a tuple, the first element is the solution array x
    result = lsqr(A_aug, b_aug)
    m_hat: np.ndarray = result[0]

    return m_hat


def solve_bayesian(dataset: SyntheticDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Path B: Bayesian Posterior Estimation
    Returns:
        m_post: Posterior mean estimate (500,)
        C_post: Posterior covariance (500, 500)
    """
    config = dataset.config

    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    A = sp.vstack([G, I_w], format="csr")
    b = np.concatenate([dataset.d_insar, dataset.w])

    # STEP 1: Set prior mean
    # The proposal states: "spatial average of well measurements broadcast to all voxels"
    m_0 = np.ones(config.total_voxels) * np.mean(dataset.w)

    # STEP 2: Build exponential prior covariance matrix
    p_indices = np.arange(config.total_voxels)
    pixel_indices = p_indices // config.n_layers
    layer_indices = p_indices % config.n_layers

    rows = pixel_indices // config.grid_size
    cols = pixel_indices % config.grid_size

    coords = np.column_stack((rows, cols, layer_indices))

    # Calculate pairwise distances
    distances = cdist(coords, coords, metric="euclidean")

    # C_m = sigma^2 * exp(-r / ell)
    C_m = config.prior_variance * np.exp(-distances / config.ell)

    # STEP 3: Build diagonal noise covariance matrix
    var_insar = np.ones(config.n_insar_obs) * (config.sigma_insar**2)
    var_well = np.ones(config.n_well_obs) * (config.sigma_well**2)
    C_n = np.diag(np.concatenate([var_insar, var_well]))

    # STEP 4: Compute Woodbury gain matrix
    A_dense = A.toarray()
    temp = A_dense @ C_m @ A_dense.T + C_n

    temp_inv = np.linalg.inv(temp)
    K = C_m @ A_dense.T @ temp_inv

    # STEP 5: Compute posterior mean
    residual = b - A_dense @ m_0
    m_post = m_0 + K @ residual

    # STEP 6: Compute posterior covariance
    C_post = C_m - K @ A_dense @ C_m

    return m_post, C_post


def solve_tikhonov_nnls(dataset: SyntheticDataset) -> np.ndarray:
    """
    Path A (Enhanced): Tikhonov Regularization with Non-Negativity Constraints.
    Minimizes: ||A*m - b||^2 + lam * ||L*m||^2 subject to m >= 0.
    """
    config = dataset.config

    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    L = build_L_matrix(config)

    A = sp.vstack([G, I_w], format="csr")
    b = np.concatenate([dataset.d_insar, dataset.w])

    sqrt_lam = np.sqrt(config.lam)
    A_aug = sp.vstack([A, sqrt_lam * L], format="csr")
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])

    # solve using lsq_linear for bounds (0, inf)
    result = lsq_linear(A_aug, b_aug, bounds=(0, np.inf))
    m_hat: np.ndarray = result.x

    return m_hat


def solve_bayesian_anisotropic(dataset: SyntheticDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Path B (Enhanced): Bayesian Posterior Estimation with Anisotropic Covariance Kernel.
    """
    config = dataset.config

    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    A = sp.vstack([G, I_w], format="csr")
    b = np.concatenate([dataset.d_insar, dataset.w])

    m_0 = np.ones(config.total_voxels) * np.mean(dataset.w)

    p_indices = np.arange(config.total_voxels)
    pixel_indices = p_indices // config.n_layers
    layer_indices = p_indices % config.n_layers

    rows = pixel_indices // config.grid_size
    cols = pixel_indices % config.grid_size

    # Anisotropic adjustment: scale coordinates based on correlation lengths
    scaled_rows = rows / config.ell
    scaled_cols = cols / config.ell
    scaled_layers = layer_indices / config.ell_z

    coords = np.column_stack((scaled_rows, scaled_cols, scaled_layers))
    distances = cdist(coords, coords, metric="euclidean")

    C_m = config.prior_variance * np.exp(-distances)

    var_insar = np.ones(config.n_insar_obs) * (config.sigma_insar**2)
    var_well = np.ones(config.n_well_obs) * (config.sigma_well**2)
    C_n = np.diag(np.concatenate([var_insar, var_well]))

    A_dense = A.toarray()
    temp = A_dense @ C_m @ A_dense.T + C_n
    temp_inv = np.linalg.inv(temp)
    K = C_m @ A_dense.T @ temp_inv

    residual = b - A_dense @ m_0
    m_post = m_0 + K @ residual
    C_post = C_m - K @ A_dense @ C_m

    return m_post, C_post


def solve_bayesian_depth_prior(dataset: SyntheticDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Path B (Enhanced): Bayesian Estimation with Depth-Dependent Priors.
    Also includes Anisotropic Covariance.
    """
    config = dataset.config

    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    A = sp.vstack([G, I_w], format="csr")
    b = np.concatenate([dataset.d_insar, dataset.w])

    # Compute depth-dependent prior mean
    # We have dataset.w structured as [well1_L1, well1_L2, ..., well5_L5]
    w_reshaped = dataset.w.reshape(-1, config.n_layers)  # (n_wells, n_layers)
    layer_means = np.mean(w_reshaped, axis=0)  # shape (n_layers,)

    p_indices = np.arange(config.total_voxels)
    layer_indices = p_indices % config.n_layers

    # Broadcast layer_means to all voxels based on their layer
    m_0 = layer_means[layer_indices]

    # Use anisotropic covariance
    pixel_indices = p_indices // config.n_layers
    rows = pixel_indices // config.grid_size
    cols = pixel_indices % config.grid_size

    scaled_rows = rows / config.ell
    scaled_cols = cols / config.ell
    scaled_layers = layer_indices / config.ell_z

    coords = np.column_stack((scaled_rows, scaled_cols, scaled_layers))
    distances = cdist(coords, coords, metric="euclidean")

    C_m = config.prior_variance * np.exp(-distances)

    var_insar = np.ones(config.n_insar_obs) * (config.sigma_insar**2)
    var_well = np.ones(config.n_well_obs) * (config.sigma_well**2)
    C_n = np.diag(np.concatenate([var_insar, var_well]))

    A_dense = A.toarray()
    temp = A_dense @ C_m @ A_dense.T + C_n
    temp_inv = np.linalg.inv(temp)
    K = C_m @ A_dense.T @ temp_inv

    residual = b - A_dense @ m_0
    m_post = m_0 + K @ residual
    C_post = C_m - K @ A_dense @ C_m

    return m_post, C_post
