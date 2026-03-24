import numpy as np
from scipy.sparse import csr_array

from src.config import SystemConfig


def build_G_matrix(config: SystemConfig) -> csr_array:
    """
    Build the InSAR forward operator G.

    Implements the discrete integral equation:
        (G @ m)_i = sum_j(f_ij * delta_z)
    where f_ij is the layer-compaction value (mm when delta_z=1.0) of layer j
    beneath pixel i and delta_z is the layer weight (default 1.0).
    The product G @ m yields the predicted displacement from 0–300 m layers only —
    NOT total surface displacement. InSAR may exceed G @ m in areas with deep
    compaction below 300 m.

    When config.insar_pixel_indices is None (default), builds one row per pixel
    in sequential order — the standard toy-problem path (fast CSR construction).
    When set, builds only the rows for those specific pixel indices, enabling
    sparse InSAR observation support for real-data inversions.

    Returns a sparse matrix of shape (n_insar_obs, total_voxels).
    """
    n_cols = config.total_voxels

    if config.insar_pixel_indices is None:
        # Fast path: sequential pixels, compact CSR representation.
        n_rows = config.n_insar_obs  # == n_pixels
        nnz = n_rows * config.n_layers
        data = np.full(nnz, config.delta_z, dtype=np.float64)
        indices = np.arange(nnz, dtype=np.int32)
        indptr = np.arange(0, nnz + 1, config.n_layers, dtype=np.int32)
        return csr_array((data, indices, indptr), shape=(n_rows, n_cols))

    # Sparse path: build COO from the explicit pixel index list.
    pixel_arr = np.asarray(config.insar_pixel_indices, dtype=np.int32)
    n_rows = len(pixel_arr)
    layer_offsets = np.arange(config.n_layers, dtype=np.int32)

    row_idx = np.repeat(np.arange(n_rows, dtype=np.int32), config.n_layers)
    col_idx = (
        np.repeat(pixel_arr * config.n_layers, config.n_layers) +
        np.tile(layer_offsets, n_rows)
    )
    data = np.full(n_rows * config.n_layers, config.delta_z, dtype=np.float64)

    return csr_array((data, (row_idx, col_idx)), shape=(n_rows, n_cols))


def build_I_wells(config: SystemConfig) -> csr_array:
    """
    Build the well selection matrix I_wells.
    Returns a sparse matrix of shape (n_well_obs, total_voxels).
    Each row picks out exactly one voxel measured by a well.
    """
    n_rows = config.n_well_obs
    n_cols = config.total_voxels

    data = np.ones(n_rows, dtype=np.float64)

    # The columns correspond to the specific voxels
    # For each well index w, the voxels are w*L to w*L + L - 1
    indices = np.zeros(n_rows, dtype=np.int32)
    row_idx = 0
    for w in config.well_indices:
        start_col = w * config.n_layers
        for l in range(config.n_layers):
            indices[row_idx] = start_col + l
            row_idx += 1

    indptr = np.arange(n_rows + 1, dtype=np.int32)

    return csr_array((data, indices, indptr), shape=(n_rows, n_cols))


def build_L_matrix(config: SystemConfig) -> csr_array:
    """
    Build the first-order finite-difference operator L along the model vector.
    Returns a sparse matrix of shape (total_voxels - 1, total_voxels).
    """
    n_rows = config.total_voxels - 1
    n_cols = config.total_voxels

    # Each row has exactly two non-zero entries: 1 at (p, p) and -1 at (p, p+1)
    nnz = n_rows * 2
    data = np.zeros(nnz, dtype=np.float64)
    indices = np.zeros(nnz, dtype=np.int32)

    # Fill data and indices
    data[0::2] = 1.0
    data[1::2] = -1.0

    indices[0::2] = np.arange(n_rows)
    indices[1::2] = np.arange(1, n_cols)

    indptr = np.arange(0, nnz + 1, 2, dtype=np.int32)

    L = csr_array((data, indices, indptr), shape=(n_rows, n_cols))

    # Zero out rows that cross pixel boundaries: row r connects voxel r to r+1.
    # When r = k*n_layers - 1 (last layer of pixel k), voxel r and r+1 belong
    # to different horizontal pixels — this difference is physically meaningless.
    n_layers = config.n_layers
    boundary_rows = np.arange(n_layers - 1, n_rows, n_layers, dtype=np.int32)
    if boundary_rows.size > 0:
        # Convert to lil for efficient row zeroing, then back to csr
        L_lil = L.tolil()
        for r in boundary_rows:
            L_lil[r] = 0
        L = L_lil.tocsr()
        L.eliminate_zeros()

    return L
