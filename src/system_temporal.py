import numpy as np
import scipy.sparse as sp

from src.config import SystemConfig


def build_block_diagonal_operator(base_matrix: sp.csr_matrix, n_epochs: int) -> sp.csr_matrix:
    """
    Takes a single-epoch operator (like G or I_w) and block-diagonalizes it
    across T epochs.
    """
    # scipy.sparse.block_diag takes a list of matrices
    matrices = [base_matrix for _ in range(n_epochs)]
    return sp.block_diag(matrices, format="csr")


def build_Lt_matrix(config: SystemConfig) -> sp.csr_matrix:
    """
    Builds the temporal finite-difference operator L_t.
    Penalizes the difference between m_{i, t} and m_{i, t+1}.

    Returns a sparse matrix of shape (M * (T-1), M * T)
    where M is total_voxels and T is n_epochs.
    """
    M = config.total_voxels
    T = config.n_epochs

    n_rows = M * (T - 1)
    n_cols = M * T

    # For each row, we have a +1 and a -1
    # row r corresponds to voxel v at time transition t to t+1
    # r = t * M + v
    # m_col1 = t * M + v
    # m_col2 = (t + 1) * M + v

    nnz = n_rows * 2
    data = np.zeros(nnz, dtype=np.float64)
    indices = np.zeros(nnz, dtype=np.int32)

    # We want to minimize (m_{t+1} - m_t) -> so weights are -1 for t, +1 for t+1
    data[0::2] = -1.0
    data[1::2] = 1.0

    # Col indices
    # First column in pair is m_t
    indices[0::2] = np.arange(0, n_rows)
    # Second column in pair is m_{t+1}
    indices[1::2] = np.arange(M, n_cols)

    indptr = np.arange(0, nnz + 1, 2, dtype=np.int32)

    return sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
