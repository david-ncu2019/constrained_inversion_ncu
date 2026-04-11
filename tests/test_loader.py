"""Tests for src/loader.py using the real my_input_data/ directory."""

from pathlib import Path

import numpy as np
import pytest

from src.loader import (
    build_real_dataset,
    coord_to_pixel_index,
    get_grid_specs,
    load_station_coords,
)

DATA_DIR = Path("my_input_data")


@pytest.fixture(scope="module")
def grid_specs():  # type: ignore[return]
    return get_grid_specs(DATA_DIR / "grid_pnt_CRFP_500m_vert_IDW_v1.nc")


def test_get_grid_specs(grid_specs: dict) -> None:  # type: ignore[type-arg]
    """NetCDF template must report the expected grid dimensions and depths."""
    assert grid_specs["grid_rows"] == 153
    assert grid_specs["grid_cols"] == 117

    depths = list(grid_specs["depths"])
    assert depths[0] == 0
    assert depths[-1] == 300
    assert len(depths) == 31
    # Depths must be at 10 m intervals
    assert all(depths[i + 1] - depths[i] == 10 for i in range(len(depths) - 1))


def test_coord_to_pixel_index(grid_specs: dict) -> None:  # type: ignore[type-arg]
    """ANHE station coordinate must map to a pixel within the 153×117 grid."""
    x_coords = grid_specs["x_coords"]
    y_coords = grid_specs["y_coords"]

    # ANHE: X_TWD97 = 179539.20, Y_TWD97 = 2602035.47
    pixel_index, grid_row, grid_col = coord_to_pixel_index(
        179539.20, 2602035.47, x_coords, y_coords
    )

    assert 0 <= grid_row < 153
    assert 0 <= grid_col < 117
    assert pixel_index == grid_row * 117 + grid_col


def test_load_station_coords(grid_specs: dict) -> None:  # type: ignore[type-arg]
    """Station coordinate loader must return at least 31 rows with valid pixel indices."""
    coords_path = DATA_DIR / "mlcw_station_coordinates.csv"
    df = load_station_coords(
        coords_path, grid_specs["x_coords"], grid_specs["y_coords"]
    )

    assert len(df) >= 31  # data-driven: actual count depends on available stations
    assert "pixel_index" in df.columns
    assert "grid_row" in df.columns
    assert "grid_col" in df.columns

    n_total_pixels = 153 * 117
    assert (df["pixel_index"] >= 0).all()
    assert (df["pixel_index"] < n_total_pixels).all()


def test_build_real_dataset_shapes() -> None:
    """Loader must return arrays with shapes consistent with the config."""
    dataset, config, meta = build_real_dataset(
        DATA_DIR,
        month_start=0,
        month_end=None,  # load all available months
        grid_metrics_file="grid_pnt_CRFP_500m_vert_IDW_v1.nc",
    )

    n_stations = config.n_pixels  # data-driven: actual station count from CSV files
    n_epochs = config.n_epochs    # determined from available columns (e.g. 83)

    assert n_stations >= 31       # at least 31 stations expected; may be more

    # Number of valid MLCW depth levels (expect 29: depths 10-290 m)
    n_layers = config.n_layers
    assert n_layers > 0
    assert all(d > 0 for d in meta["valid_depths_m"])  # depth=0 (InSAR surface) excluded
    assert all(d <= 300 for d in meta["valid_depths_m"])  # depths within instrument range

    # Config consistency
    assert config.n_epochs == n_epochs
    assert config.total_voxels == n_stations * n_layers

    # Dataset array shapes
    assert dataset.d_insar.shape == (n_stations * n_epochs,)
    assert dataset.w.shape == (n_stations * n_layers * n_epochs,)
    assert dataset.m_true.shape == (n_stations * n_layers,)

    # No NaN after interpolation
    assert not np.any(np.isnan(dataset.d_insar))
    assert not np.any(np.isnan(dataset.w))

    # Metadata completeness
    assert len(meta["station_names"]) == n_stations
    assert len(meta["full_grid_pixel_indices"]) == n_stations
    assert len(meta["month_labels"]) == n_epochs
    assert meta["grid_rows"] == 153
    assert meta["grid_cols"] == 117
