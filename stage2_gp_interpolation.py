"""
stage2_gp_interpolation.py — Gaussian Process interpolation of station-level
depth weights to the full 500 m grid.

Usage
-----
  uv run python stage2_gp_interpolation.py [options]

Options
-------
  --inversion-npz PATH   Path to real_m_est.npz (default: output/real_m_est.npz)
  --grid-nc PATH         Path to 500 m grid NetCDF (default:
                         my_input_data/grid_pnt_CRFP_500m_vert_IDW_v1.nc)
  --output-nc PATH       Output NetCDF path (default:
                         output/depth_weights_gp_grid.nc)
  --n-restarts INT       GP optimiser restarts per layer (default: 10)
  --std-threshold FLOAT  Flag pixels with GP std > this value as low-confidence
                         (default: 0.3)
  --no-cumsum            Do NOT accumulate the InSAR time series in the grid
                         NetCDF before applying weights.  Use only if the
                         NetCDF already contains cumulative displacements.

Inputs
------
  output/real_m_est.npz
      Keys used: depth_weights (n_stations, n_layers), x_twd97, y_twd97,
                 valid_depths_m, month_labels, cumulative_space.

  my_input_data/grid_pnt_CRFP_500m_vert_IDW_v1.nc
      Dimensions: Time(132), Depth(31), Y(153), X(117).
      Variable: displacement[Time, Depth=0, Y, X] = monthly incremental
      InSAR surface displacement (mm).  Depth=0 slice is used.

Outputs
-------
  output/depth_weights_gp_grid.nc
      Dimensions: layer(n_layers), Y(153), X(117), time(132).
      Variables:
        depth_weight_mean  (layer, Y, X)    — GP predicted mean weight
        depth_weight_std   (layer, Y, X)    — GP predicted std (uncertainty)
        layer_compaction   (time, layer, Y, X) — mm
        compaction_std     (time, layer, Y, X) — mm
        deep_residual      (time, Y, X)     — mm
        deep_residual_std  (time, Y, X)     — mm (propagated uncertainty)
        low_confidence     (Y, X)           — bool mask (1 = low confidence)
        insar_cum          (time, Y, X)     — cumulative InSAR used (mm)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GP interpolation of depth weights to the 500 m grid."
    )
    p.add_argument(
        "--inversion-npz", default="output/real_m_est.npz", type=Path
    )
    p.add_argument(
        "--grid-nc",
        default="my_input_data/grid_pnt_CRFP_500m_vert_IDW_v1.nc",
        type=Path,
    )
    p.add_argument(
        "--output-nc", default="output/depth_weights_gp_grid.nc", type=Path
    )
    p.add_argument(
        "--n-restarts", default=10, type=int,
        help="GP optimiser restarts per layer.",
    )
    p.add_argument(
        "--std-threshold", default=0.3, type=float,
        help="Flag pixels with max GP std above this as low-confidence.",
    )
    p.add_argument(
        "--no-cumsum", action="store_true", default=False,
        help="Skip cumsum on the grid NetCDF InSAR time series.",
    )
    p.add_argument(
        "--spatial-cv-csv", default=None, type=Path,
        help="Path to loso_per_layer.csv from cv_spatial_gp.py. "
             "If provided, spatial CV RMSE is included in total uncertainty.",
    )
    p.add_argument(
        "--temporal-cv-csv", default=None, type=Path,
        help="Path to temporal_cv_per_layer.csv from cv_temporal_forward.py. "
             "If provided, temporal CV RMSE is included in total uncertainty.",
    )
    return p.parse_args()


def load_inversion_results(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str], bool]:
    """
    Load station-level inversion output.

    Returns
    -------
    depth_weights : np.ndarray, shape (n_stations, n_layers)
    x_twd97       : np.ndarray, shape (n_stations,)
    y_twd97       : np.ndarray, shape (n_stations,)
    valid_depths_m: list of int
    month_labels  : list of str
    cumulative_space : bool
    """
    data = np.load(npz_path, allow_pickle=True)
    depth_weights = data["depth_weights"]          # (n_stations, n_layers)
    x_twd97 = data["x_twd97"].astype(np.float64)
    y_twd97 = data["y_twd97"].astype(np.float64)
    valid_depths_m = data["valid_depths_m"].tolist()
    month_labels = data["month_labels"].tolist()
    cumulative_space = bool(data["cumulative_space"])
    return depth_weights, x_twd97, y_twd97, valid_depths_m, month_labels, cumulative_space


def load_grid_insar(
    nc_path: Path, apply_cumsum: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load InSAR surface displacement from the 500 m grid NetCDF.

    Parameters
    ----------
    nc_path : Path
        Path to grid_pnt_CRFP_500m_vert_IDW_v1.nc.
    apply_cumsum : bool
        If True, accumulate the monthly increments to cumulative displacement.

    Returns
    -------
    insar_surface : np.ndarray, shape (n_time, n_y, n_x)
        Cumulative (or incremental) surface displacement in mm.
        NaN where no InSAR coverage.
    x_coords : np.ndarray, shape (n_x,)
    y_coords : np.ndarray, shape (n_y,)
    time_labels : np.ndarray of str, shape (n_time,)
    """
    ds = xr.open_dataset(nc_path)
    # Depth=0 slice is the InSAR surface displacement
    disp = ds["displacement"].sel(Depth=0).values  # (n_time, n_y, n_x), float32
    disp = disp.astype(np.float64)

    x_coords = ds["X"].values.copy()
    y_coords = ds["Y"].values.copy()
    time_labels = ds["month_label"].values.astype(str)
    ds.close()

    # Negate: raw data negative = subsidence; make subsidence positive
    disp = -disp

    if apply_cumsum:
        # Cumulative from first month; NaN propagates forward — fill with 0
        # for cumsum then restore NaN mask
        nan_mask = np.isnan(disp)
        disp_filled = np.where(nan_mask, 0.0, disp)
        disp_cum = np.cumsum(disp_filled, axis=0)
        # Re-apply NaN where the original was NaN throughout the record
        all_nan = nan_mask.all(axis=0)  # pixels with no valid data at any time
        disp_cum[:, all_nan] = np.nan
        return disp_cum, x_coords, y_coords, time_labels
    else:
        return disp, x_coords, y_coords, time_labels


def fit_gp_per_layer(
    depth_weights: np.ndarray,
    x_train_km: np.ndarray,
    y_train_km: np.ndarray,
    x_pred_km: np.ndarray,
    y_pred_km: np.ndarray,
    n_restarts: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit independent GPs for each depth layer and predict at grid pixels.

    Parameters
    ----------
    depth_weights : np.ndarray, shape (n_stations, n_layers)
    x_train_km, y_train_km : np.ndarray, shape (n_stations,) — km
    x_pred_km, y_pred_km   : np.ndarray, shape (n_pixels,)  — km
    n_restarts : int

    Returns
    -------
    mean_grid : np.ndarray, shape (n_layers, n_pixels)
    std_grid  : np.ndarray, shape (n_layers, n_pixels)
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    n_layers = depth_weights.shape[1]
    n_pixels = len(x_pred_km)

    X_train = np.column_stack([x_train_km, y_train_km])
    X_pred = np.column_stack([x_pred_km, y_pred_km])

    mean_grid = np.full((n_layers, n_pixels), np.nan)
    std_grid = np.full((n_layers, n_pixels), np.nan)

    for l in tqdm(range(n_layers), desc="GP fitting per layer", unit="layer"):
        y_train = depth_weights[:, l]

        # Skip layers where all weights are zero (no compaction signal)
        if np.all(y_train == 0):
            mean_grid[l] = 0.0
            std_grid[l] = 0.0
            continue

        kernel = (
            Matern(nu=2.5, length_scale=5.0, length_scale_bounds=(1.0, 50.0))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 0.1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
        )
        gp.fit(X_train, y_train)
        mu, sigma = gp.predict(X_pred, return_std=True)
        mean_grid[l] = mu
        std_grid[l] = sigma

    return mean_grid, std_grid


def normalise_weights(
    mean_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clip and normalise GP mean weights so that Σ_l mean[l, p] ≤ 1.

    Negative values are clipped to 0 (inversion artefacts).
    Pixels where the sum exceeds 1 are scaled down proportionally.

    Parameters
    ----------
    mean_grid : np.ndarray, shape (n_layers, n_pixels)

    Returns
    -------
    mean_norm : np.ndarray, shape (n_layers, n_pixels) — normalised
    scale_factor : np.ndarray, shape (n_pixels,) — 1 where sum ≤ 1, <1 otherwise
    """
    mean_clipped = np.maximum(mean_grid, 0.0)
    total = mean_clipped.sum(axis=0)               # (n_pixels,)
    scale_factor = np.where(total > 1.0, 1.0 / total, 1.0)
    mean_norm = mean_clipped * scale_factor[np.newaxis, :]
    return mean_norm, scale_factor


def assign_temporal_rmse(
    temporal_cv_df: pd.DataFrame,
    valid_depths_m: list[int],
    n_time: int,
) -> np.ndarray:
    """
    Expand per-fold temporal CV RMSE to the full time axis.

    For time index t:
      t < 48  → fold 1 RMSE (best available proxy for early extrapolation)
      48 ≤ t < 60 → fold 1 RMSE
      60 ≤ t < 72 → fold 2 RMSE
      72 ≤ t      → fold 3 RMSE  (also used for all post-training-period times)

    Parameters
    ----------
    temporal_cv_df : pd.DataFrame
        Columns: fold, depth_m, rmse, ...  (from temporal_cv_per_layer.csv)
    valid_depths_m : list[int], length n_layers
    n_time : int  — total number of time steps (e.g. 132)

    Returns
    -------
    sigma_temporal : np.ndarray, shape (n_time, n_layers) — in mm
    """
    fold_boundaries = {
        1: (0, 59),    # fold 1 covers time 0–59
        2: (60, 71),   # fold 2 covers time 60–71
        3: (72, None), # fold 3 covers 72 onward
    }
    n_layers = len(valid_depths_m)
    depth_to_idx = {d: i for i, d in enumerate(valid_depths_m)}

    sigma_temporal = np.full((n_time, n_layers), np.nan)

    for fold_id, (t_start, t_end) in fold_boundaries.items():
        fold_rows = temporal_cv_df[temporal_cv_df["fold"] == fold_id]
        rmse_arr = np.zeros(n_layers)
        for _, row in fold_rows.iterrows():
            d = int(row["depth_m"])
            if d in depth_to_idx:
                val = float(row["rmse"])
                rmse_arr[depth_to_idx[d]] = 0.0 if np.isnan(val) else val

        t_slice = slice(t_start, t_end + 1 if t_end is not None else None)
        sigma_temporal[t_slice, :] = rmse_arr[np.newaxis, :]

    return sigma_temporal


def load_cv_errors(
    spatial_csv: Path | None,
    temporal_csv: Path | None,
    valid_depths_m: list[int],
    n_time: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load spatial and temporal CV RMSE outputs and align them to valid_depths_m.

    Parameters
    ----------
    spatial_csv : Path or None
        loso_per_layer.csv from cv_spatial_gp.py.
        Expected columns: depth_m, rmse, mae, bias, std_error.
    temporal_csv : Path or None
        temporal_cv_per_layer.csv from cv_temporal_forward.py.
        Expected columns: fold, depth_m, rmse, ...
    valid_depths_m : list[int], length n_layers
    n_time : int

    Returns
    -------
    spatial_cv_rmse  : np.ndarray shape (n_layers,) or None
        Dimensionless weight-space RMSE per layer.
    temporal_cv_sigma: np.ndarray shape (n_time, n_layers) or None
        mm-space RMSE per time step per layer.
    """
    spatial_cv_rmse = None
    if spatial_csv is not None and spatial_csv.exists():
        df = pd.read_csv(spatial_csv)
        depth_to_idx = {d: i for i, d in enumerate(valid_depths_m)}
        rmse_arr = np.zeros(len(valid_depths_m))
        for _, row in df.iterrows():
            d = int(row["depth_m"])
            if d in depth_to_idx:
                val = float(row["rmse"])
                rmse_arr[depth_to_idx[d]] = 0.0 if np.isnan(val) else val
        spatial_cv_rmse = rmse_arr
        print(f"  Loaded spatial CV RMSE from: {spatial_csv}")
        print(f"    Mean RMSE across layers: {rmse_arr.mean():.5f} (dimensionless)")

    temporal_cv_sigma = None
    if temporal_csv is not None and temporal_csv.exists():
        df = pd.read_csv(temporal_csv)
        temporal_cv_sigma = assign_temporal_rmse(df, valid_depths_m, n_time)
        print(f"  Loaded temporal CV RMSE from: {temporal_csv}")
        print(f"    Mean RMSE across layers and time: {np.nanmean(temporal_cv_sigma):.3f} mm")

    return spatial_cv_rmse, temporal_cv_sigma


def build_output_dataset(
    mean_grid: np.ndarray,
    std_grid: np.ndarray,
    insar_cum: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    time_labels: np.ndarray,
    valid_depths_m: list[int],
    std_threshold: float,
    valid_mask: np.ndarray,
    spatial_cv_rmse: np.ndarray | None = None,
    temporal_cv_sigma: np.ndarray | None = None,
) -> xr.Dataset:
    """
    Assemble the output xarray Dataset.

    Parameters
    ----------
    mean_grid : np.ndarray, shape (n_layers, n_y*n_x) — flattened pixels
    std_grid  : np.ndarray, shape (n_layers, n_y*n_x)
    insar_cum : np.ndarray, shape (n_time, n_y, n_x)
    x_coords  : np.ndarray, shape (n_x,)
    y_coords  : np.ndarray, shape (n_y,)
    time_labels : np.ndarray of str, shape (n_time,)
    valid_depths_m : list of int, length n_layers
    std_threshold : float
    valid_mask : np.ndarray bool, shape (n_y*n_x,) — pixels with InSAR coverage

    Returns
    -------
    xr.Dataset
    """
    n_y = len(y_coords)
    n_x = len(x_coords)
    n_layers = len(valid_depths_m)
    n_time = insar_cum.shape[0]
    n_pixels = n_y * n_x

    # Reshape mean/std to (n_layers, n_y, n_x) — NaN outside valid_mask
    mean_2d = np.full((n_layers, n_pixels), np.nan)
    std_2d = np.full((n_layers, n_pixels), np.nan)
    mean_2d[:, valid_mask] = mean_grid
    std_2d[:, valid_mask] = std_grid

    mean_3d = mean_2d.reshape(n_layers, n_y, n_x)
    std_3d = std_2d.reshape(n_layers, n_y, n_x)

    # Layer compaction and uncertainty: (n_time, n_layers, n_y, n_x)
    # insar_cum has shape (n_time, n_y, n_x); mean_3d has (n_layers, n_y, n_x)
    layer_compaction = (
        mean_3d[np.newaxis, :, :, :] * insar_cum[:, np.newaxis, :, :]
    )
    compaction_std = (
        std_3d[np.newaxis, :, :, :] * np.abs(insar_cum[:, np.newaxis, :, :])
    )

    # Deep residual: InSAR_cum - Σ_l layer_compaction
    deep_residual = insar_cum - layer_compaction.sum(axis=1)  # (n_time, n_y, n_x)

    # Deep residual uncertainty (propagation assuming layer independence)
    deep_residual_std = np.sqrt(
        (compaction_std**2).sum(axis=1)
    )  # (n_time, n_y, n_x)

    # Combined total uncertainty: σ_total² = σ_gp² + σ_spatial_CV² + σ_temporal_CV²
    compaction_total_std: np.ndarray | None = None
    if spatial_cv_rmse is not None and temporal_cv_sigma is not None:
        # spatial_cv_rmse: (n_layers,) — dimensionless weight-space RMSE
        # Convert to mm: sigma_spatial_mm[t,l,y,x] = rmse[l] * |InSAR_cum[t,y,x]|
        sigma_spatial_mm = (
            spatial_cv_rmse[np.newaxis, :, np.newaxis, np.newaxis]
            * np.abs(insar_cum[:, np.newaxis, :, :])
        )  # (n_time, n_layers, n_y, n_x)

        # temporal_cv_sigma: (n_time, n_layers) — already in mm
        sigma_temporal_mm = (
            temporal_cv_sigma[:, :, np.newaxis, np.newaxis]
            * np.ones((n_y, n_x), dtype=np.float32)
        )  # (n_time, n_layers, n_y, n_x)

        # compaction_std is sigma_gp (already in mm): (n_time, n_layers, n_y, n_x)
        compaction_total_std = np.sqrt(
            compaction_std**2 + sigma_spatial_mm**2 + sigma_temporal_mm**2
        ).astype(np.float32)

    # Low-confidence mask: any layer with high std at that pixel
    low_confidence = (std_3d.max(axis=0) > std_threshold)  # (n_y, n_x)

    depths_coord = np.array(valid_depths_m, dtype=np.float32)

    ds = xr.Dataset(
        {
            "depth_weight_mean": xr.DataArray(
                mean_3d.astype(np.float32),
                dims=["layer", "Y", "X"],
                attrs={
                    "long_name": "GP-predicted mean depth weight",
                    "description": (
                        "Fractional contribution of each 10 m depth layer to "
                        "total InSAR cumulative surface displacement. "
                        "Normalised so that sum over layers ≤ 1."
                    ),
                    "units": "dimensionless",
                },
            ),
            "depth_weight_std": xr.DataArray(
                std_3d.astype(np.float32),
                dims=["layer", "Y", "X"],
                attrs={
                    "long_name": "GP predictive standard deviation of depth weight",
                    "units": "dimensionless",
                },
            ),
            "layer_compaction": xr.DataArray(
                layer_compaction.astype(np.float32),
                dims=["time", "layer", "Y", "X"],
                attrs={
                    "long_name": "Predicted cumulative compaction per depth layer",
                    "units": "mm",
                },
            ),
            "compaction_std": xr.DataArray(
                compaction_std.astype(np.float32),
                dims=["time", "layer", "Y", "X"],
                attrs={
                    "long_name": "Uncertainty (1σ) of predicted cumulative compaction",
                    "units": "mm",
                },
            ),
            "deep_residual": xr.DataArray(
                deep_residual.astype(np.float32),
                dims=["time", "Y", "X"],
                attrs={
                    "long_name": "Deep residual compaction (below 300 m)",
                    "description": (
                        "InSAR_cum - sum(layer_compaction). Positive = unexplained "
                        "subsidence attributed to compaction below 300 m."
                    ),
                    "units": "mm",
                },
            ),
            "deep_residual_std": xr.DataArray(
                deep_residual_std.astype(np.float32),
                dims=["time", "Y", "X"],
                attrs={
                    "long_name": "Uncertainty (1σ) of deep residual",
                    "units": "mm",
                },
            ),
            "low_confidence": xr.DataArray(
                low_confidence.astype(np.int8),
                dims=["Y", "X"],
                attrs={
                    "long_name": "Low-confidence pixel mask",
                    "description": (
                        f"1 where max GP std across layers > {std_threshold:.2f}. "
                        "Indicates extrapolation artefacts far from MLCW stations."
                    ),
                },
            ),
            "insar_cum": xr.DataArray(
                insar_cum.astype(np.float32),
                dims=["time", "Y", "X"],
                attrs={
                    "long_name": "Cumulative InSAR surface displacement used",
                    "units": "mm",
                },
            ),
            **(
                {
                    "compaction_total_std": xr.DataArray(
                        compaction_total_std,
                        dims=["time", "layer", "Y", "X"],
                        attrs={
                            "long_name": "Combined total uncertainty (1σ) of predicted layer compaction",
                            "description": (
                                "sqrt(sigma_gp^2 + sigma_spatial_CV^2 + sigma_temporal_CV^2). "
                                "sigma_gp = depth_weight_std * |InSAR_cum|; "
                                "sigma_spatial_CV = LOSO RMSE (dimensionless) * |InSAR_cum|; "
                                "sigma_temporal_CV = forward-chaining CV RMSE (mm)."
                            ),
                            "units": "mm",
                        },
                    )
                }
                if compaction_total_std is not None
                else {}
            ),
        },
        coords={
            "X": xr.DataArray(x_coords, dims=["X"], attrs={"units": "m", "crs": "EPSG:3826"}),
            "Y": xr.DataArray(y_coords, dims=["Y"], attrs={"units": "m", "crs": "EPSG:3826"}),
            "layer": xr.DataArray(depths_coord, dims=["layer"], attrs={"units": "m", "long_name": "Depth (top of layer)"}),
            "time": xr.DataArray(time_labels, dims=["time"]),
        },
        attrs={
            "description": "GP-interpolated depth weights and layer-wise compaction predictions",
            "CRS": "EPSG:3826 (TWD97 TM2)",
            "grid_resolution_m": 500,
            "created_by": "stage2_gp_interpolation.py",
        },
    )
    return ds


def main() -> None:
    args = parse_args()
    args.output_nc.parent.mkdir(parents=True, exist_ok=True)

    # ── Load inversion results ────────────────────────────────────────────
    print(f"Loading inversion results from: {args.inversion_npz}")
    (
        depth_weights,
        x_twd97,
        y_twd97,
        valid_depths_m,
        month_labels,
        cumulative_space,
    ) = load_inversion_results(args.inversion_npz)

    n_stations, n_layers = depth_weights.shape
    print(f"  Stations : {n_stations}")
    print(f"  Layers   : {n_layers}  (depths: {valid_depths_m[0]}–{valid_depths_m[-1]} m)")
    print(f"  Cumulative space: {cumulative_space}")

    # ── Load grid InSAR ───────────────────────────────────────────────────
    apply_cumsum = not args.no_cumsum
    print(f"\nLoading grid InSAR from: {args.grid_nc}")
    print(f"  Apply cumsum to InSAR: {apply_cumsum}")
    insar_cum, x_coords, y_coords, time_labels = load_grid_insar(
        args.grid_nc, apply_cumsum=apply_cumsum
    )
    n_time, n_y, n_x = insar_cum.shape
    n_pixels = n_y * n_x
    print(f"  Grid shape: {n_y} × {n_x} = {n_pixels} pixels, {n_time} time steps")

    # ── Identify pixels with any valid InSAR coverage ────────────────────
    valid_mask = ~np.isnan(insar_cum).all(axis=0).flatten()  # (n_pixels,)
    n_valid = valid_mask.sum()
    print(f"  Valid pixels: {n_valid} / {n_pixels} ({100*n_valid/n_pixels:.1f}%)")

    # ── Build prediction coordinates (km) ────────────────────────────────
    xx, yy = np.meshgrid(x_coords, y_coords)              # (n_y, n_x)
    x_all_km = (xx.flatten() / 1000.0)[valid_mask]
    y_all_km = (yy.flatten() / 1000.0)[valid_mask]

    x_train_km = x_twd97 / 1000.0
    y_train_km = y_twd97 / 1000.0

    # ── Fit GPs per layer ─────────────────────────────────────────────────
    print(f"\nFitting {n_layers} GPs (n_restarts={args.n_restarts}) ...")
    mean_valid, std_valid = fit_gp_per_layer(
        depth_weights=depth_weights,
        x_train_km=x_train_km,
        y_train_km=y_train_km,
        x_pred_km=x_all_km,
        y_pred_km=y_all_km,
        n_restarts=args.n_restarts,
    )
    # mean_valid, std_valid: (n_layers, n_valid_pixels)

    # ── Normalise weights (sum ≤ 1 per pixel) ────────────────────────────
    mean_norm, scale_factor = normalise_weights(mean_valid)
    n_rescaled = (scale_factor < 1.0).sum()
    print(f"\nWeight normalisation: {n_rescaled} / {n_valid} valid pixels rescaled (sum > 1)")

    # ── Load CV error estimates (optional) ───────────────────────────────
    spatial_cv_rmse, temporal_cv_sigma = load_cv_errors(
        spatial_csv=args.spatial_cv_csv,
        temporal_csv=args.temporal_cv_csv,
        valid_depths_m=valid_depths_m,
        n_time=n_time,
    )
    if spatial_cv_rmse is not None and temporal_cv_sigma is not None:
        print("  Combined uncertainty (compaction_total_std) will be included in output.")
    else:
        missing = []
        if spatial_cv_rmse is None:
            missing.append("--spatial-cv-csv")
        if temporal_cv_sigma is None:
            missing.append("--temporal-cv-csv")
        print(
            f"  NOTE: {' and '.join(missing)} not provided. "
            "compaction_total_std will NOT be included. "
            "Run cv_spatial_gp.py and cv_temporal_forward.py first."
        )

    # ── Assemble output dataset ───────────────────────────────────────────
    print("Assembling output dataset ...")
    ds = build_output_dataset(
        mean_grid=mean_norm,
        std_grid=std_valid,
        insar_cum=insar_cum,
        x_coords=x_coords,
        y_coords=y_coords,
        time_labels=time_labels,
        valid_depths_m=valid_depths_m,
        std_threshold=args.std_threshold,
        valid_mask=valid_mask,
        spatial_cv_rmse=spatial_cv_rmse,
        temporal_cv_sigma=temporal_cv_sigma,
    )

    low_conf_count = int(ds["low_confidence"].values.sum())
    print(f"  Low-confidence pixels (std > {args.std_threshold}): {low_conf_count}")

    # Deep residual sanity check
    dr = ds["deep_residual"].values
    n_negative = int((dr < -0.1).sum())
    print(f"  Deep residual negative cells (<-0.1 mm): {n_negative}")
    if n_negative > 0:
        print(
            "  WARNING: Negative deep residual detected. This indicates that "
            "predicted 0–300 m compaction exceeds InSAR in some pixels. "
            "Check depth_weight normalisation and InSAR sign convention."
        )

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving to: {args.output_nc}")
    ds.to_netcdf(args.output_nc)
    print("Done.")


if __name__ == "__main__":
    main()
