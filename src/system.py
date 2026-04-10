import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from src.config import SystemConfig


def build_G_matrix(config: SystemConfig) -> csr_matrix:
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
        return csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))

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

    return csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, n_cols))


def build_I_wells(config: SystemConfig) -> csr_matrix:
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

    return csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))


def build_L_matrix(config: SystemConfig) -> csr_matrix:
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

    L = csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))

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


def build_L_spatial(
    config: SystemConfig,
    x_twd97: np.ndarray,
    y_twd97: np.ndarray,
) -> csr_matrix:
    """
    Build a geographically meaningful spatial smoothness operator.

    Connects station pairs within ``config.spatial_dist_threshold_m`` metres
    using distance-weighted finite differences.  Each edge (i, j) contributes
    one row per depth layer, penalising the difference between the model values
    at the two stations for that layer.

    This replaces the sequential ``build_L_matrix`` for the real-data path,
    where stations are not ordered geographically.

    Parameters
    ----------
    config : SystemConfig
        Must have ``n_layers`` and ``total_voxels`` set correctly for the
        real-data inversion (``n_pixels = n_stations``).
    x_twd97 : np.ndarray, shape (n_stations,)
        Easting coordinates in TWD97 metres.
    y_twd97 : np.ndarray, shape (n_stations,)
        Northing coordinates in TWD97 metres.

    Returns
    -------
    csr_matrix, shape (n_edges * n_layers, total_voxels)
        Row ``edge_idx * n_layers + l`` enforces similarity between layer ``l``
        of station ``i`` and layer ``l`` of station ``j``, weighted by the
        inverse distance between the two stations (normalised so the maximum
        weight is 1).
    """
    n_stations = config.n_pixels
    n_layers = config.n_layers
    threshold = config.spatial_dist_threshold_m

    coords = np.column_stack([x_twd97, y_twd97])           # (n_stations, 2)
    dist_matrix = cdist(coords, coords, metric="euclidean")  # (n_stations, n_stations)

    # Collect unique edges (i < j) within the distance threshold
    rows_i, cols_j = np.where(
        (dist_matrix > 0) & (dist_matrix <= threshold)
    )
    mask = rows_i < cols_j
    edges_i = rows_i[mask]
    edges_j = cols_j[mask]
    n_edges = len(edges_i)

    if n_edges == 0:
        # No edges found — return a zero-row matrix so block_diagonal works
        return csr_matrix((0, config.total_voxels), dtype=np.float64)

    # Distance weights: inverse distance, normalised to [0, 1]
    edge_dists = dist_matrix[edges_i, edges_j]
    raw_weights = 1.0 / edge_dists
    weights = raw_weights / raw_weights.max()

    # Build COO arrays for all edges × all layers
    n_rows = n_edges * n_layers
    n_cols = config.total_voxels

    # For edge k and layer l: row = k * n_layers + l
    # col_pos = edges_i[k] * n_layers + l  → weight +w_k
    # col_neg = edges_j[k] * n_layers + l  → weight -w_k
    layer_idx = np.arange(n_layers, dtype=np.int32)

    # Repeat each edge index for every layer
    edge_rep = np.repeat(np.arange(n_edges, dtype=np.int32), n_layers)
    layer_rep = np.tile(layer_idx, n_edges)
    weight_rep = np.repeat(weights, n_layers)

    row_idx = edge_rep * n_layers + layer_rep

    col_pos = edges_i[edge_rep] * n_layers + layer_rep
    col_neg = edges_j[edge_rep] * n_layers + layer_rep

    # Interleave positive and negative entries
    total_nnz = n_rows * 2
    coo_row = np.empty(total_nnz, dtype=np.int32)
    coo_col = np.empty(total_nnz, dtype=np.int32)
    coo_data = np.empty(total_nnz, dtype=np.float64)

    coo_row[0::2] = row_idx
    coo_row[1::2] = row_idx
    coo_col[0::2] = col_pos
    coo_col[1::2] = col_neg
    coo_data[0::2] = weight_rep
    coo_data[1::2] = -weight_rep

    from scipy.sparse import coo_matrix
    L_spatial = coo_matrix(
        (coo_data, (coo_row, coo_col)), shape=(n_rows, n_cols)
    ).tocsr()
    L_spatial.eliminate_zeros()
    return L_spatial
