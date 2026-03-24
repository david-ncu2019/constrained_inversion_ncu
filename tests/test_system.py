import numpy as np
import pytest
from scipy.sparse import csr_array

from src.config import SystemConfig
from src.system import build_G_matrix, build_I_wells, build_L_matrix


@pytest.fixture
def config() -> SystemConfig:
    return SystemConfig()


def test_build_G_matrix(config: SystemConfig) -> None:
    G = build_G_matrix(config)

    # Check type and shape
    assert isinstance(G, csr_array)
    assert G.shape == (100, 500)

    # Each row should sum to n_layers * delta_z (layer thickness weighted sum)
    expected_row_sum = config.n_layers * config.delta_z
    row_sums = G.sum(axis=1)
    assert np.all(row_sums == expected_row_sum)

    # Check the first row explicitly
    first_row = G[0].toarray().flatten()
    assert np.sum(first_row) == expected_row_sum
    assert np.all(first_row[0:5] == config.delta_z)
    assert np.all(first_row[5:] == 0)

    # Check the last row explicitly
    last_row = G[-1].toarray().flatten()
    assert np.sum(last_row) == expected_row_sum
    assert np.all(last_row[-5:] == config.delta_z)
    assert np.all(last_row[:-5] == 0)


def test_build_I_wells(config: SystemConfig) -> None:
    I_w = build_I_wells(config)

    # Check type and shape
    assert isinstance(I_w, csr_array)
    assert I_w.shape == (25, 500)

    # Each row should have exactly 1 one
    row_sums = I_w.sum(axis=1)
    assert np.all(row_sums == 1)

    # Ensure the 1s are in the right columns for the first well (index 11)
    # The first well corresponds to rows 0-4 of I_w
    # Pixel 11's layers are at columns 11*5 = 55 to 59
    well_1_cols = [np.nonzero(I_w[i].toarray().flatten())[0][0] for i in range(5)]
    assert well_1_cols == [55, 56, 57, 58, 59]


def test_build_L_matrix(config: SystemConfig) -> None:
    L = build_L_matrix(config)

    # Check type and shape
    assert isinstance(L, csr_array)
    assert L.shape == (499, 500)

    # Check first row [1, -1, 0, ...]
    first_row = L[0].toarray().flatten()
    assert first_row[0] == 1
    assert first_row[1] == -1
    assert np.sum(np.abs(first_row)) == 2


def test_build_L_matrix_boundary_zeroing(config: SystemConfig) -> None:
    """Verify that L zeroes out rows at pixel boundaries."""
    L = build_L_matrix(config)
    L_dense = L.toarray()
    n_layers = config.n_layers
    n_rows = L.shape[0]

    # Boundary rows: indices [4, 9, 14, ..., 494] for n_layers=5
    boundary_rows = np.arange(n_layers - 1, n_rows, n_layers)
    for r in boundary_rows:
        assert np.all(L_dense[r] == 0), f"Boundary row {r} should be all-zero"

    # Non-boundary rows must have exactly 2 nonzero entries summing to 0
    all_rows = set(range(n_rows))
    interior_rows = sorted(all_rows - set(boundary_rows))
    for r in interior_rows:
        row = L_dense[r]
        nonzero = row[row != 0]
        assert len(nonzero) == 2, f"Row {r} should have 2 nonzero entries"
        assert np.isclose(nonzero.sum(), 0), f"Row {r} entries should sum to 0"


def test_build_L_matrix_nondefault_layers() -> None:
    """Verify boundary zeroing adapts to non-default n_layers."""
    cfg = SystemConfig(grid_size=4, n_layers=7, well_indices=(0, 5))
    L = build_L_matrix(cfg)
    L_dense = L.toarray()
    n_rows = L.shape[0]

    boundary_rows = np.arange(cfg.n_layers - 1, n_rows, cfg.n_layers)
    # Expected boundaries: [6, 13, 20, ...]
    assert boundary_rows[0] == 6
    assert boundary_rows[1] == 13

    for r in boundary_rows:
        assert np.all(L_dense[r] == 0), f"Boundary row {r} should be all-zero"


def test_build_G_matrix_nonunit_delta_z() -> None:
    """Verify G scales correctly with non-unit delta_z."""
    cfg = SystemConfig(grid_size=4, n_layers=5, well_indices=(0, 5), delta_z=5.0)
    G = build_G_matrix(cfg)

    # Row sums should equal n_layers * delta_z
    expected_sum = cfg.n_layers * cfg.delta_z  # 5 * 5.0 = 25.0
    row_sums = np.asarray(G.sum(axis=1)).flatten()
    np.testing.assert_allclose(row_sums, expected_sum)

    # Forward consistency: uniform model of 1.0 mm/m should give 25 mm surface disp.
    m_uniform = np.ones(cfg.total_voxels)
    d = G @ m_uniform
    np.testing.assert_allclose(d, expected_sum)
