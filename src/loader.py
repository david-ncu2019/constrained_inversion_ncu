"""
Loader for real CRFP MLCW + InSAR data.

Input format (standard, from my_input_data/):

  CSV_files/{STATION}_insar_mlcw.csv
      One file per MLCW station.
      Rows: Depth = 0, 10, 20, ..., 300 m (31 depths at 10 m intervals).
      Columns: Depth, Month_001, Month_002, ..., Month_083.
      Values: monthly INCREMENTAL displacement in mm (NOT cumulative).
      Depth = 0     → InSAR-derived surface deformation at that station.
      Depth = 10-290 → MLCW layer compaction of each 10 m layer (mm).
      Depth = 300   → all NaN; excluded automatically.
      Month_N       → N months from January 2015 (Month_0 = Jan 2015, baseline = 0).

  mlcw_station_coordinates.csv
      Columns: Ename, Code, X_TWD97, Y_TWD97.
      One row per station (31 total).

  grid_pnt_datacube_500m.nc
      Spatial output template: 153 rows × 117 cols, 500 m resolution, TWD97.
      Currently all NaN — used only to read grid metadata (X, Y coordinates).

Physical units contract:
  delta_z = 1.0  →  G entries = 1.0  →  G @ m = sum(mm_layer_k for 0–300 m) = partial surface mm
  (0–300 m only). InSAR surface displacement may exceed G @ m due to deep compaction.
  No unit conversion is performed; CSV values flow directly into d_insar and w.

Inversion grid:
  For the initial implementation the 31 station locations form the inversion grid.
  Each station becomes pixel index 0..30 in the model (sequential).
  The output NetCDF grid geometry is preserved in the returned metadata dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from src.config import SyntheticDataset, SystemConfig
import configparser


def parse_dataset_config(config_path: Path | str | None) -> dict[str, Any]:
    """Extract and cast [Dataset] parameters from pipeline_config.ini."""
    if not config_path: return {}
    p = Path(config_path)
    if not p.exists(): return {}

    cfg = configparser.ConfigParser()
    cfg.read(p, encoding="utf-8")
    if not cfg.has_section("Dataset"): return {}

    kwargs = {}
    ds = dict(cfg.items("Dataset"))
    str_keys = [
        "grid_metrics_file", "station_coords_file", "csv_dir_name", 
        "csv_name_pattern", "x_col", "y_col", "depth_col", "month_col_prefix"
    ]
    for k in str_keys:
        if k in ds: kwargs[k] = ds[k].strip()

    if "insar_surface_depth" in ds:
        kwargs["insar_surface_depth"] = int(ds["insar_surface_depth"])
    if "min_insar_fraction" in ds:
        kwargs["min_insar_fraction"] = float(ds["min_insar_fraction"])

    return kwargs

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def get_grid_specs(
    nc_path: Path,
    x_dim: str = "X",
    y_dim: str = "Y",
    depth_dim: str = "Depth",
) -> dict[str, Any]:
    """
    Read spatial metadata from the NetCDF output template.

    Parameters
    ----------
    nc_path : Path
        Path to grid_pnt_datacube_500m.nc.

    Returns
    -------
    dict with keys:
        x_coords  : np.ndarray, shape (n_cols,) — TWD97 easting values
        y_coords  : np.ndarray, shape (n_rows,) — TWD97 northing values
        depths    : np.ndarray — depth levels [0, 10, ..., 300]
        grid_rows : int — number of Y grid points (153)
        grid_cols : int — number of X grid points (117)
    """
    ds = xr.open_dataset(nc_path)
    specs: dict[str, Any] = {
        "x_coords": ds[x_dim].values.copy(),
        "y_coords": ds[y_dim].values.copy(),
        "depths": ds[depth_dim].values.copy(),
        "grid_rows": int(ds.sizes[y_dim]),
        "grid_cols": int(ds.sizes[x_dim]),
    }
    ds.close()
    return specs


def coord_to_pixel_index(
    x: float,
    y: float,
    x_coords: npt.NDArray[np.float64],
    y_coords: npt.NDArray[np.float64],
) -> tuple[int, int, int]:
    """
    Map one TWD97 coordinate pair to the nearest grid pixel.

    Returns
    -------
    (pixel_index, grid_row, grid_col)
        pixel_index = grid_row * len(x_coords) + grid_col
    """
    col = int(np.argmin(np.abs(x_coords - x)))
    row = int(np.argmin(np.abs(y_coords - y)))
    return row * len(x_coords) + col, row, col


def load_station_coords(
    coords_path: Path,
    x_coords: npt.NDArray[np.float64],
    y_coords: npt.NDArray[np.float64],
    x_col: str = "X_TWD97",
    y_col: str = "Y_TWD97",
) -> pd.DataFrame:
    """
    Load station coordinates and compute nearest-grid-point pixel indices.

    Returns
    -------
    pd.DataFrame with columns:
        Ename, Code, X_TWD97, Y_TWD97, pixel_index, grid_row, grid_col
    """
    df = pd.read_csv(coords_path)
        
    pixel_indices, grid_rows, grid_cols = [], [], []
    for _, row in df.iterrows():
        pidx, gr, gc = coord_to_pixel_index(
            float(row[x_col]), float(row[y_col]), x_coords, y_coords
        )
        pixel_indices.append(pidx)
        grid_rows.append(gr)
        grid_cols.append(gc)
    df["pixel_index"] = pixel_indices
    df["grid_row"] = grid_rows
    df["grid_col"] = grid_cols
    return df


def find_valid_mlcw_depths(
    csv_dir: Path,
    station_names: list[str],
    month_col_prefix: str = "Month_",
    csv_name_pattern: str = "{name}_insar_mlcw.csv",
    depth_col: str = "Depth",
) -> list[int]:
    """
    Find depth levels (depth > 0) that have at least one non-NaN value
    across all stations and all months.  Depths that are entirely NaN
    (e.g., depth = 300 m) are excluded automatically.

    Returns
    -------
    Sorted list of valid MLCW depth values (integers, in metres).
    """
    valid: set[int] = set()
    for name in station_names:
        csv_path = csv_dir / csv_name_pattern.format(name=name)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).set_index(depth_col)
        month_cols = [c for c in df.columns if str(c).startswith(month_col_prefix)]
        for depth_val, row in df.iterrows():
            depth = int(depth_val)
            if depth == 0:
                continue
            if not row[month_cols].isna().all():
                valid.add(depth)
    return sorted(valid)


def _interpolate_series(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Fill NaN gaps in a 1-D time series using linear interpolation.

    - Interior NaN values are filled by linear interpolation.
    - Leading / trailing NaN values are filled by the nearest valid value
      (forward / backward extrapolation clamped to boundary).
    - If the entire series is NaN, returns zeros.
    """
    result = np.array(values, dtype=np.float64)
    valid_mask = ~np.isnan(result)
    if not np.any(valid_mask):
        return np.zeros_like(result)
    valid_x = np.where(valid_mask)[0]
    all_x = np.arange(len(result))
    # np.interp clamps to boundary values for out-of-range points.
    result = np.interp(all_x, valid_x, result[valid_mask])
    return result


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def build_real_dataset(
    data_dir: str | Path,
    month_start: int = 0,
    month_end: int | None = None,
    lam: float = 1e-2,
    lam_t: float = 1.0,
    sigma_insar: float = 3.0,
    sigma_well: float = 1.0,
    cumulate: bool = True,
    grid_metrics_file: str = "grid_pnt_datacube_500m.nc",
    station_coords_file: str = "mlcw_station_coordinates.csv",
    csv_dir_name: str = "CSV_files",
    csv_name_pattern: str = "{name}_insar_mlcw.csv",
    x_col: str = "X_TWD97",
    y_col: str = "Y_TWD97",
    depth_col: str = "Depth",
    month_col_prefix: str = "Month_",
    insar_surface_depth: int = 0,
    min_insar_fraction: float = 0.50,
) -> tuple[SyntheticDataset, SystemConfig, dict[str, Any]]:
    """
    Load real CRFP monitoring data into a SyntheticDataset for inversion.

    The 31 MLCW station locations form the inversion grid.  Each station is
    assigned a sequential pixel index (0..30) in the model.  The returned
    config therefore has n_pixels = n_stations (31) and n_layers = number
    of valid MLCW depth levels (typically 30, covering 10–300 m).

    Month columns are auto-detected from the first station CSV (e.g.,
    'Month_001', 'Month_002', ..., 'Month_083').  month_start / month_end
    are 0-based indices into the sorted column list.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing CSV_files/, mlcw_station_coordinates.csv, and
        grid_pnt_datacube_500m.nc.
    month_start : int
        0-based index of the first month column to load (default: 0 = first).
    month_end : int or None
        0-based index of the last month column to load (inclusive).
        None (default) = last available column.
    lam : float
        Spatial Tikhonov regularisation parameter.
    lam_t : float
        Temporal regularisation parameter.
    sigma_insar : float
        InSAR observation noise standard deviation (mm).
    sigma_well : float
        Well observation noise standard deviation (mm).
    cumulate : bool
        If True (default), convert incremental monthly values to cumulative
        displacements from the reference date before inversion.  Cumulative
        space is required to enforce the physical inequality constraint
        Σ_l m_cum[t,s,l] ≤ InSAR_cum[t,s] reliably at every epoch.
        Set to False only for diagnostic comparison with the incremental path.

    Returns
    -------
    dataset : SyntheticDataset
        Ready for solve_joint_spacetime() or solve_independent_epochs().
        m_true is set to zeros (ground truth unknown for real data).
    config : SystemConfig
        Configuration for the inversion.
    metadata : dict
        Auxiliary information for output / visualisation:
            station_names            : list[str]
            x_twd97, y_twd97         : list[float]  TWD97 coordinates
            full_grid_pixel_indices  : list[int]  indices in the 153×117 output grid
            grid_rows, grid_cols     : int  output grid dimensions
            valid_depths_m           : list[int]  MLCW depth levels used (m)
            month_labels             : list[str]  actual column names loaded
            month_start, month_end   : int
    """
    data_dir = Path(data_dir)
    nc_path = data_dir / grid_metrics_file
    coords_path = data_dir / station_coords_file
    csv_dir = data_dir / csv_dir_name

    # ── 1. Spatial grid metadata from NetCDF template ────────────────────
    grid = get_grid_specs(nc_path)
    x_coords: npt.NDArray[np.float64] = grid["x_coords"]
    y_coords: npt.NDArray[np.float64] = grid["y_coords"]

    # ── 2. Station coordinates → nearest output-grid pixel ───────────────
    stations = load_station_coords(coords_path, x_coords, y_coords, x_col=x_col, y_col=y_col)
    station_names: list[str] = stations["Ename"].tolist()

    # ── 2b. Exclude stations with insufficient InSAR temporal coverage ────
    # Stations with < 50% valid InSAR months are excluded: boundary-clamping
    # extrapolation for the missing epochs has no physical basis.
    _all_month_sample = pd.read_csv(
        csv_dir / csv_name_pattern.format(name=station_names[0]), index_col=0
    )
    _n_total_months = len([c for c in _all_month_sample.columns if str(c).startswith(month_col_prefix)])

    def _count_valid_insar(name: str) -> int:
        p = csv_dir / csv_name_pattern.format(name=name)
        if not p.exists():
            return 0
        _df = pd.read_csv(p, index_col=0)
        _month_cols = [c for c in _df.columns if str(c).startswith(month_col_prefix)]
        if insar_surface_depth not in _df.index:
            return 0
        return int(_df.loc[insar_surface_depth, _month_cols].notna().sum())

    _valid_counts = {name: _count_valid_insar(name) for name in station_names}
    _excluded = [n for n, c in _valid_counts.items()
                 if c < min_insar_fraction * _n_total_months]
    if _excluded:
        print(f"[loader] Excluding {len(_excluded)} stations with < {int(min_insar_fraction*100)}% InSAR coverage: {_excluded}")
    stations = stations[~stations["Ename"].isin(_excluded)].reset_index(drop=True)
    station_names = stations["Ename"].tolist()
    n_stations = len(stations)

    # ── 3. Valid MLCW depth levels ────────────────────────────────────────
    valid_depths = find_valid_mlcw_depths(
        csv_dir, station_names, month_col_prefix, csv_name_pattern, depth_col
    )
    n_layers = len(valid_depths)

    # ── 4. Auto-detect month columns ──────────────────────────────────────
    # Column names are zero-padded (e.g., 'Month_001' ... 'Month_083').
    # Auto-detect from the first station CSV to avoid hardcoding the format.
    sample_csv_path = csv_dir / csv_name_pattern.format(name=station_names[0])
    sample_df = pd.read_csv(sample_csv_path)
    all_month_cols = sorted([str(c) for c in sample_df.columns if str(c).startswith(month_col_prefix)])
    month_end_idx = month_end if month_end is not None else len(all_month_cols) - 1
    month_cols = all_month_cols[month_start : month_end_idx + 1]
    n_epochs = len(month_cols)

    # ── 5. Build SystemConfig ─────────────────────────────────────────────
    # Stations become sequential pixels 0..n_stations-1 in the inversion.
    # Use grid_rows=n_stations, grid_cols=1 so that n_pixels = n_stations
    # (not n_stations², which grid_size=n_stations would produce).
    # delta_z = 1.0: G entries equal 1.0, so G @ m = sum(mm_layer) = mm ✓
    config = SystemConfig(
        grid_rows=n_stations,
        grid_cols=1,
        n_layers=n_layers,
        well_indices=tuple(range(n_stations)),
        delta_z=1.0,
        n_epochs=n_epochs,
        lam=lam,
        lam_t=lam_t,
        sigma_insar=sigma_insar,
        sigma_well=sigma_well,
    )

    # ── 6. Build raw data matrices ────────────────────────────────────────
    # d_insar_matrix[s, t] = surface displacement at station s, epoch t (mm)
    # w_matrix[s*n_layers + l, t] = layer compaction, station s, layer l, epoch t (mm)
    d_insar_matrix = np.full((n_stations, n_epochs), np.nan)
    w_matrix = np.full((n_stations * n_layers, n_epochs), np.nan)

    for s_idx, (_, srow) in enumerate(stations.iterrows()):
        name = str(srow["Ename"])
        csv_path = csv_dir / csv_name_pattern.format(name=name)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).set_index(depth_col)

        # InSAR: depth = insar_surface_depth
        if insar_surface_depth in df.index:
            for t_idx, mc in enumerate(month_cols):
                if mc in df.columns:
                    val = df.loc[insar_surface_depth, mc]
                    if not pd.isna(val):
                        d_insar_matrix[s_idx, t_idx] = float(val)

        # MLCW: valid depth levels (mm, no unit conversion required)
        for l_idx, depth in enumerate(valid_depths):
            if depth not in df.index:
                continue
            well_row = s_idx * n_layers + l_idx
            for t_idx, mc in enumerate(month_cols):
                if mc in df.columns:
                    val = df.loc[depth, mc]
                    if not pd.isna(val):
                        w_matrix[well_row, t_idx] = float(val)

    # ── 7. NaN interpolation (per station / depth time series) ───────────
    for s_idx in range(n_stations):
        d_insar_matrix[s_idx, :] = _interpolate_series(d_insar_matrix[s_idx, :])
        for l_idx in range(n_layers):
            row = s_idx * n_layers + l_idx
            w_matrix[row, :] = _interpolate_series(w_matrix[row, :])

    # ── 7b. Cumulative conversion (optional) ─────────────────────────────
    # Convert incremental monthly values to cumulative displacements from
    # the reference date.  In cumulative space the inequality constraint
    # Σ_l m_cum[t,s,l] ≤ InSAR_cum[t,s] is physically reliable at every
    # epoch, avoiding spurious violations caused by measurement noise that
    # occur when comparing incremental (monthly) values.
    if cumulate:
        d_insar_matrix = np.cumsum(d_insar_matrix, axis=1)
        w_matrix = np.cumsum(w_matrix, axis=1)

    # ── 8. Sign convention and flatten ───────────────────────────────────
    # Real CRFP data convention: negative = subsidence (ground moves down).
    # Code convention: positive m means compaction (layer shortens, surface
    # subsides).  The forward model is d_insar = G @ m, so positive m must
    # produce positive d_insar.  Multiply all observations by −1 to reconcile:
    #   raw < 0 (subsidence)  →  negated > 0  →  consistent with m ≥ 0.
    # For rebound months (raw > 0), the negated value < 0 falls below the
    # non-negativity bound and the solver assigns m ≈ 0 (no active compaction).
    d_insar_matrix = -d_insar_matrix
    w_matrix = -w_matrix

    # Layout required by solve_joint_spacetime / solve_independent_epochs:
    #   d_insar: [epoch0_data (n_stations,), epoch1_data, ..., epochT-1_data]
    #   w:       [epoch0_data (n_well_obs,), epoch1_data, ..., epochT-1_data]
    # d_insar_matrix has shape (n_stations, n_epochs); transposing and flattening
    # gives the correct epoch-major order.
    d_insar_flat = d_insar_matrix.T.flatten().astype(np.float64)
    w_flat = w_matrix.T.flatten().astype(np.float64)

    dataset = SyntheticDataset(
        config=config,
        m_true=np.zeros(config.total_voxels, dtype=np.float64),
        d_insar=d_insar_flat,
        w=w_flat,
    )

    metadata: dict[str, Any] = {
        "station_names": station_names,
        "x_twd97": stations[x_col].tolist(),
        "y_twd97": stations[y_col].tolist(),
        "full_grid_pixel_indices": stations["pixel_index"].tolist(),
        "grid_rows": int(grid["grid_rows"]),
        "grid_cols": int(grid["grid_cols"]),
        "valid_depths_m": valid_depths,
        "month_labels": month_cols,
        "month_start": month_start,
        "month_end": month_end_idx,
        "cumulative_space": cumulate,
    }

    return dataset, config, metadata
