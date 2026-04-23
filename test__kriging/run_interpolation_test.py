"""
Synthetic Benchmark for RotatedGPR and AnisotropicKriging Engines.
=================================================================
Reads scenario definitions from ``scenarios.yaml``.
Skips scenarios whose diagnostics folder already exists (use --force to re-run).

Usage:
    python run_interpolation_test.py                  # run NEW scenarios only
    python run_interpolation_test.py --force           # re-run everything
    python run_interpolation_test.py --only S7,S8      # run specific scenarios
"""
import argparse, os, sys, warnings, yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.spatial_solvers import RotatedGPR, AnisotropicKriging
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from diagnostics import (
    compute_empirical_variogram, plot_variogram, plot_directional_variogram,
    perform_gpr_kfold_cv, perform_kriging_kfold_cv,
    plot_cv_dashboard, plot_anisotropy_ellipse,
)


# ====================================================================
# Ground truth generator
# ====================================================================
def generate_gaussian_field(nx, ny, xsiz, ysiz, angle_deg,
                            range_major, range_minor, psill, nugget, seed=42):
    """FFT spectral simulation with exact Gaussian covariance."""
    rng = np.random.RandomState(seed)
    dx = np.fft.fftfreq(nx, d=1.0) * nx * xsiz
    dy = np.fft.fftfreq(ny, d=1.0) * ny * ysiz
    DX, DY = np.meshgrid(dx, dy)
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    DXr =  DX * cos_t + DY * sin_t
    DYr = -DX * sin_t + DY * cos_t
    h_sq = (DXr / range_major) ** 2 + (DYr / range_minor) ** 2
    C_lag = psill * np.exp(-3.0 * h_sq)
    S = np.real(np.fft.fft2(C_lag))
    S = np.maximum(S, 0.0)
    noise = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    field = np.real(np.fft.ifft2(np.sqrt(S) * noise))
    field = (field - field.mean()) / field.std() * np.sqrt(psill)
    field += rng.normal(0, np.sqrt(nugget), (ny, nx))
    return field


# ====================================================================
# Single-scenario runner
# ====================================================================
def run_scenario(sc_name, params, grid_cfg, caps):
    """Run one scenario: generate data, fit engines, produce diagnostics."""

    nx, ny   = grid_cfg["nx"], grid_cfg["ny"]
    xsiz     = grid_cfg["xsiz"]
    ysiz     = grid_cfg["ysiz"]
    xmn, ymn = grid_cfg["xmn"], grid_cfg["ymn"]

    x_coords = np.linspace(xmn, xmn + (nx - 1) * xsiz, nx)
    y_coords = np.linspace(ymn, ymn + (ny - 1) * ysiz, ny)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

    true_angle = params["angle"]
    true_rmaj  = params["range_major"]
    true_rmin  = params["range_minor"]
    true_psill = params["psill"]
    true_nugget = params["nugget"]
    true_ratio = true_rmaj / true_rmin
    sample_frac = params["sample_frac"]
    seed = params.get("seed", 42)

    out_dir = f"{sc_name}_diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Generate ground truth -----------------------------------------------
    truth = generate_gaussian_field(
        nx, ny, xsiz, ysiz, true_angle, true_rmaj, true_rmin,
        true_psill, true_nugget, seed=seed,
    )
    print(f"  Field var = {truth.var():.3f}  "
          f"(expected ~ {true_psill + true_nugget:.3f})")

    ds = xr.Dataset(
        {"ground_truth": (["y", "x"], truth)},
        coords={"x": ("x", x_coords, {"units": "m"}),
                "y": ("y", y_coords, {"units": "m"})},
    )
    ds.to_netcdf(f"ground_truth_{sc_name}.nc")

    # 2. Subsample -----------------------------------------------------------
    flat_truth = truth.flatten()
    flat_X, flat_Y = X_grid.flatten(), Y_grid.flatten()
    n_total = nx * ny
    n_sample = max(int(sample_frac * n_total), 10)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    X_sample = np.column_stack((flat_X[idx], flat_Y[idx]))
    y_sample = flat_truth[idx]
    print(f"  Sampled {n_sample} / {n_total}  ({sample_frac*100:.0f}%)")

    MAX_GPR  = caps.get("gpr", 1500)
    MAX_KRIG = caps.get("kriging", 2000)

    if n_sample > MAX_GPR:
        gi = rng.choice(n_sample, MAX_GPR, replace=False)
        X_gpr, y_gpr = X_sample[gi], y_sample[gi]
        print(f"  GPR subsample: {MAX_GPR} pts")
    else:
        X_gpr, y_gpr = X_sample, y_sample

    if n_sample > MAX_KRIG:
        ki = rng.choice(n_sample, MAX_KRIG, replace=False)
        X_krig, y_krig = X_sample[ki], y_sample[ki]
        print(f"  Kriging subsample: {MAX_KRIG} pts")
    else:
        X_krig, y_krig = X_sample, y_sample

    # 3. Empirical variogram (adaptive lag selection) --------------------------
    print("  Computing empirical variogram ...")
    true_p = {"psill": true_psill, "range_major": true_rmaj,
              "nugget": true_nugget}
    vario_omni = compute_empirical_variogram(X_krig, y_krig)  # fully adaptive
    ap = vario_omni.get("auto_params", {})
    print(f"    Auto lags: n_lags={ap.get('n_lags')}  "
          f"lag_width={ap.get('lag_width', 0):.1f} m  "
          f"max_lag={ap.get('max_lag', 0):.0f} m  "
          f"median_NN={ap.get('median_nn', 0):.1f} m  "
          f"avg_pairs/bin={ap.get('avg_pairs_per_bin', 0):.0f}")
    if ap.get("method_notes", "") != "all defaults applied":
        print(f"    Notes: {ap['method_notes']}")
    plot_variogram(vario_omni, true_params=true_p,
                   scenario_name=sc_name, engine_name="Empirical",
                   save_path=os.path.join(out_dir, "variogram_omni.png"))

    dirs = list(range(0, 180, 15))  # 15-degree intervals: 0, 15, ..., 165
    vario_dir = compute_empirical_variogram(
        X_krig, y_krig, directions=dirs, tol_angle=15.0)  # also adaptive
    plot_directional_variogram(
        vario_dir, true_params=true_p, scenario_name=sc_name,
        save_path=os.path.join(out_dir, "variogram_directional.png"))

    # 4. Fit RotatedGPR ------------------------------------------------------
    print("  Fitting RotatedGPR ...")
    base_kernel = C(1.0, (1e-3, 1e3)) * RBF([100.0, 100.0], (1e-1, 1e4))
    gpr = RotatedGPR(
        kernel=base_kernel, angle_search_method="bounded",
        max_anisotropy=12.0, n_restarts_optimizer=1, random_state=42,
    )
    gpr.fit(X_gpr, y_gpr)

    # 5. Fit AnisotropicKriging ----------------------------------------------
    print("  Fitting AnisotropicKriging ...")
    ak = AnisotropicKriging(
        n_trials=80, n_splits=5, random_state=42, max_anisotropy=12.0,
    )
    ak.fit(X_krig, y_krig)

    # 6. Full-grid predictions -----------------------------------------------
    X_full = np.column_stack((flat_X, flat_Y))
    y_gpr_pred, y_gpr_std = gpr.predict(X_full, return_std=True)
    y_ak_pred, y_ak_std = ak.predict(X_full, return_std=True)

    # 7. Metrics -------------------------------------------------------------
    def _rmss(yt, yp, ys):
        return np.sqrt(np.mean(((yt - yp) / (ys + 1e-8)) ** 2))

    rmse_gpr = np.sqrt(np.mean((flat_truth - y_gpr_pred) ** 2))
    mae_gpr  = np.mean(np.abs(flat_truth - y_gpr_pred))
    rmss_gpr = _rmss(flat_truth, y_gpr_pred, y_gpr_std)

    rmse_ak  = np.sqrt(np.mean((flat_truth - y_ak_pred) ** 2))
    mae_ak   = np.mean(np.abs(flat_truth - y_ak_pred))
    rmss_ak  = _rmss(flat_truth, y_ak_pred, y_ak_std)

    gpr_p = gpr.get_kernel_params()
    ak_p  = ak.get_kernel_params()

    # 8. Variogram with fitted Kriging curve ---------------------------------
    ak_fitted_p = {"psill": ak_p.get("psill", 1),
                   "range": ak_p.get("range", 200),
                   "nugget": ak_p.get("nugget", 0)}
    plot_variogram(vario_omni, true_params=true_p,
                   fitted_params=ak_fitted_p,
                   engine_name="Kriging", scenario_name=sc_name,
                   save_path=os.path.join(out_dir, "variogram_kriging.png"))

    # 9. Cross-validation ----------------------------------------------------
    print("  Running GPR cross-validation ...")
    n_cv_folds = min(5, n_sample // 3)  # adjust for very sparse data
    cv_gpr = perform_gpr_kfold_cv(gpr, X_gpr, y_gpr, n_folds=n_cv_folds)
    plot_cv_dashboard(cv_gpr, engine_name="RotatedGPR",
                      scenario_name=sc_name,
                      save_path=os.path.join(out_dir, "cv_gpr.png"))

    print("  Running Kriging cross-validation ...")
    cv_ak = perform_kriging_kfold_cv(ak, X_krig, y_krig, n_folds=n_cv_folds)
    plot_cv_dashboard(cv_ak, engine_name="AnisotropicKriging",
                      scenario_name=sc_name,
                      save_path=os.path.join(out_dir, "cv_kriging.png"))

    # 10. Anisotropy ellipse -------------------------------------------------
    plot_anisotropy_ellipse(
        gpr_p, true_params={"angle": true_angle, "range_major": true_rmaj,
                            "range_minor": true_rmin},
        engine_name="GPR", scenario_name=sc_name,
        save_path=os.path.join(out_dir, "ellipse_gpr.png"))

    # 11. 6-panel comparison -------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    extent = (xmn, xmn + nx * xsiz, ymn, ymn + ny * ysiz)
    for ax, data, title in [
        (axes[0, 0], truth, "Ground Truth"),
        (axes[0, 1], y_gpr_pred.reshape(ny, nx), "GPR Pred"),
        (axes[0, 2], y_ak_pred.reshape(ny, nx), "Kriging Pred"),
        (axes[1, 0], np.abs(flat_truth - y_gpr_pred).reshape(ny, nx),
         "GPR |Error|"),
        (axes[1, 1], y_gpr_std.reshape(ny, nx), "GPR StdDev"),
        (axes[1, 2], y_ak_std.reshape(ny, nx), "Kriging StdDev"),
    ]:
        cm = ("Reds" if "Error" in title
              else ("magma" if "Std" in title else "viridis"))
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cm)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    # overlay sample points on ground truth
    axes[0, 0].scatter(X_sample[:, 0], X_sample[:, 1],
                       c="red", s=3, alpha=0.5, label=f"n={n_sample}")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    plt.suptitle(f"{sc_name}  ({sample_frac*100:.0f}% = {n_sample} pts)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=150)
    plt.close()

    # 12. Print summary ------------------------------------------------------
    print(f"\n  --- Parameter Recovery ---")
    print(f"  TRUE   : Angle={true_angle:6.1f}  Ratio={true_ratio:5.2f}  "
          f"Nugget={true_nugget:.3f}  Psill={true_psill:.3f}")
    print(f"  GPR    : Angle={gpr_p['rotation_angle_deg']:6.1f}  "
          f"Ratio={gpr_p['anisotropy_ratio']:5.2f}  "
          f"Noise={gpr_p['noise_level']:.3f}")
    print(f"  KRIGING: Angle={ak_p['rotation_angle_deg']:6.1f}  "
          f"Ratio={ak_p['anisotropy_ratio']:5.2f}  "
          f"Nugget={ak_p['nugget']:.3f}  Psill={ak_p['psill']:.3f}")
    print(f"\n  --- Prediction Metrics ---")
    print(f"  GPR     -> RMSE={rmse_gpr:.4f}  MAE={mae_gpr:.4f}  "
          f"RMSS={rmss_gpr:.4f}")
    print(f"  KRIGING -> RMSE={rmse_ak:.4f}  MAE={mae_ak:.4f}  "
          f"RMSS={rmss_ak:.4f}")

    if len(cv_gpr) > 0:
        print(f"  GPR CV  -> RMSE="
              f"{np.sqrt((cv_gpr['Residual']**2).mean()):.4f}  "
              f"RMSS={np.sqrt((cv_gpr['Z_Score']**2).mean()):.4f}")
    if len(cv_ak) > 0:
        print(f"  KRIG CV -> RMSE="
              f"{np.sqrt((cv_ak['Residual']**2).mean()):.4f}  "
              f"RMSS={np.sqrt((cv_ak['Z_Score']**2).mean()):.4f}")

    print(f"  Diagnostics saved to {out_dir}/")

    return {
        "Scenario": sc_name, "Sample_Frac": sample_frac, "N_Sample": n_sample,
        "True_Angle": true_angle, "True_Ratio": true_ratio,
        "True_Nugget": true_nugget,
        "GPR_Angle": gpr_p["rotation_angle_deg"],
        "GPR_Ratio": gpr_p["anisotropy_ratio"],
        "GPR_Noise": gpr_p["noise_level"],
        "GPR_RMSE": rmse_gpr, "GPR_RMSS": rmss_gpr,
        "KRIG_Angle": ak_p["rotation_angle_deg"],
        "KRIG_Ratio": ak_p["anisotropy_ratio"],
        "KRIG_Nugget": ak_p["nugget"],
        "KRIG_RMSE": rmse_ak, "KRIG_RMSS": rmss_ak,
    }


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="Interpolation benchmark")
    parser.add_argument("--config", default="scenarios.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--force", action="store_true",
                        help="Re-run scenarios even if diagnostics exist")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated scenario prefixes to run "
                             "(e.g. S7,S8)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    grid_cfg = cfg.pop("grid")
    caps = cfg.pop("engine_caps", {"gpr": 1500, "kriging": 2000})

    # Filter scenarios
    scenario_names = [k for k in cfg.keys()]
    if args.only:
        prefixes = [p.strip() for p in args.only.split(",")]
        scenario_names = [s for s in scenario_names
                          if any(s.startswith(p) for p in prefixes)]

    summary_rows = []
    for sc_name in scenario_names:
        out_dir = f"{sc_name}_diagnostics"
        if os.path.isdir(out_dir) and not args.force:
            print(f"\nSkipping {sc_name} (folder exists, use --force)")
            continue

        print(f"\n{'='*60}")
        print(f"  {sc_name}")
        print(f"{'='*60}")

        row = run_scenario(sc_name, cfg[sc_name], grid_cfg, caps)
        summary_rows.append(row)
        print(f"  {'='*50}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_name = "benchmark_summary.csv"
        # Append if file exists
        if os.path.isfile(csv_name):
            existing = pd.read_csv(csv_name)
            df = pd.concat([existing, df], ignore_index=True)
            df.drop_duplicates(subset=["Scenario"], keep="last", inplace=True)
        df.to_csv(csv_name, index=False, float_format="%.4f")
        print(f"\nBenchmark summary -> {csv_name}")
    else:
        print("\nNo scenarios to run.")


if __name__ == "__main__":
    main()
