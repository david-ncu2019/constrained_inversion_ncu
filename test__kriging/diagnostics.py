"""
Diagnostics module for interpolation engine validation.
Provides empirical variogram, cross-validation, and publication-quality plots.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import clone

warnings.filterwarnings("ignore")


# ========================================================================
# 1. EMPIRICAL VARIOGRAM
# ========================================================================

def auto_lag_params(X, min_pairs_per_bin=30, min_lags=6, max_lags=30):
    """
    Determine adaptive variogram lag parameters from data geometry.

    Strategy
    --------
    1. **lag_width** = 2 x median nearest-neighbour distance.
       This ensures the first lag bin captures short-range structure and
       contains enough pairs for a stable semivariance estimate.
    2. **max_lag**   = min(0.5 * max_pairwise_distance, extent that
       guarantees >= min_pairs_per_bin pairs on average per bin).
    3. **n_lags**    = max_lag / lag_width, clamped to [min_lags, max_lags].

    Parameters
    ----------
    X : (n, 2) array – spatial coordinates
    min_pairs_per_bin : int – target minimum pair count per lag bin
    min_lags : int – hard lower bound on lag count
    max_lags : int – hard upper bound on lag count

    Returns
    -------
    dict with keys: n_lags, lag_width, max_lag, max_dist, median_nn,
                    n_total_pairs, method_notes (str)
    """
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    n_total_pairs = n * (n - 1) // 2

    dists_sq = squareform(pdist(X))
    max_dist = dists_sq.max()

    # Nearest-neighbour distances (excluding self)
    np.fill_diagonal(dists_sq, np.inf)
    nn_dists = dists_sq.min(axis=1)
    median_nn = float(np.median(nn_dists))

    notes = []

    # --- lag_width: 2x median nearest-neighbour ---
    lag_width = 2.0 * median_nn
    # Safety floor: at least 1% of max distance
    if lag_width < max_dist * 0.01:
        lag_width = max_dist * 0.01
        notes.append("lag_width floored to 1% of max_dist")

    # --- max_lag: half of max pairwise distance ---
    max_lag = 0.5 * max_dist

    # --- n_lags from max_lag / lag_width, then clamp ---
    n_lags = int(np.round(max_lag / lag_width))
    if n_lags < min_lags:
        n_lags = min_lags
        notes.append(f"n_lags raised to min_lags={min_lags}")
    if n_lags > max_lags:
        n_lags = max_lags
        notes.append(f"n_lags capped at max_lags={max_lags}")

    # Adjust lag_width and max_lag for consistency
    lag_width = max_lag / n_lags

    # Check pair density: if avg pairs/bin is too low, reduce n_lags
    avg_pairs = n_total_pairs / n_lags
    while avg_pairs < min_pairs_per_bin and n_lags > min_lags:
        n_lags -= 1
        lag_width = max_lag / n_lags
        avg_pairs = n_total_pairs / n_lags
    if avg_pairs < min_pairs_per_bin:
        notes.append(f"WARNING: avg {avg_pairs:.0f} pairs/bin < target {min_pairs_per_bin}")

    return {
        "n_lags": n_lags,
        "lag_width": float(lag_width),
        "max_lag": float(max_lag),
        "max_dist": float(max_dist),
        "median_nn": float(median_nn),
        "n_total_pairs": n_total_pairs,
        "avg_pairs_per_bin": float(n_total_pairs / n_lags),
        "method_notes": "; ".join(notes) if notes else "all defaults applied",
    }


def compute_empirical_variogram(X, y, n_lags=None, max_lag=None,
                                lag_width=None, max_lag_frac=None,
                                directions=None, tol_angle=22.5):
    """
    Compute omnidirectional or directional empirical semivariogram.

    Lag parameters are **adaptive by default**.  If none of n_lags /
    max_lag / lag_width / max_lag_frac are given, the function calls
    ``auto_lag_params`` to choose values based on point density and
    domain extent.  You can override any subset; the rest will be
    inferred consistently.

    Parameters
    ----------
    X : (n, 2) array  -- spatial coordinates
    y : (n,) array    -- values
    n_lags : int or None
        Number of lag bins.  Auto-determined if None.
    max_lag : float or None
        Maximum lag distance (m).  Auto-determined if None.
    lag_width : float or None
        Width of each lag bin (m).  Auto-determined if None.
    max_lag_frac : float or None
        LEGACY fallback: max lag as fraction of max pairwise distance.
        Only used when max_lag is None AND auto mode is overridden by
        providing n_lags explicitly.
    directions : list of float or None
        If None  -> omnidirectional.
        If list  -> compute for each angle (degrees CCW from +X).
    tol_angle : float -- angular tolerance (degrees) for directional bins

    Returns
    -------
    dict with keys:
        lags, semivariance, n_pairs  (arrays, length n_lags)
        lag_width, max_lag           (scalars)
        auto_params                  (dict, from auto_lag_params)
    If directional, returns list of such dicts (one per direction).
    """
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # ---- Resolve lag parameters adaptively ----
    ap = auto_lag_params(X)  # always compute for reporting

    if n_lags is None and max_lag is None and lag_width is None and max_lag_frac is None:
        # Full auto mode
        n_lags   = ap["n_lags"]
        max_lag  = ap["max_lag"]
        lag_width = ap["lag_width"]
    else:
        # Partial override
        dists_tmp = pdist(X)
        max_dist = dists_tmp.max()
        if max_lag is None:
            frac = max_lag_frac if max_lag_frac is not None else 0.5
            max_lag = max_dist * frac
        if n_lags is not None and lag_width is None:
            lag_width = max_lag / n_lags
        elif lag_width is not None and n_lags is None:
            n_lags = max(6, int(np.round(max_lag / lag_width)))
            lag_width = max_lag / n_lags   # reconcile
        elif n_lags is None and lag_width is None:
            n_lags   = ap["n_lags"]
            lag_width = max_lag / n_lags

    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    # Pairwise distances & squared differences
    dists = squareform(pdist(X))
    n = len(y)
    diff_sq = np.subtract.outer(y, y) ** 2

    def _compute_bins(angle_mask=None):
        semivar = np.zeros(n_lags)
        counts = np.zeros(n_lags, dtype=int)
        for k in range(n_lags):
            lo, hi = lag_edges[k], lag_edges[k + 1]
            mask = (dists > lo) & (dists <= hi)
            if angle_mask is not None:
                mask &= angle_mask
            np.fill_diagonal(mask, False)
            counts[k] = mask.sum() // 2
            if counts[k] > 0:
                semivar[k] = 0.5 * diff_sq[mask].sum() / mask.sum()
        return semivar, counts

    base_result = {
        "lag_width": lag_width,
        "max_lag": max_lag,
        "auto_params": ap,
    }

    if directions is None:
        sv, ct = _compute_bins()
        return {**base_result, "lags": lag_centers,
                "semivariance": sv, "n_pairs": ct}

    # Directional
    dx = X[:, 0][:, None] - X[:, 0][None, :]
    dy = X[:, 1][:, None] - X[:, 1][None, :]
    angles_deg = np.degrees(np.arctan2(dy, dx)) % 360

    results = []
    for d in directions:
        d_mod = d % 360
        d_opp = (d + 180) % 360
        ang_ok = (np.abs(angles_deg - d_mod) <= tol_angle) | \
                 (np.abs(angles_deg - d_opp) <= tol_angle) | \
                 (np.abs(angles_deg - d_mod + 360) <= tol_angle) | \
                 (np.abs(angles_deg - d_mod - 360) <= tol_angle)
        sv, ct = _compute_bins(angle_mask=ang_ok)
        results.append({
            **base_result, "direction": d,
            "lags": lag_centers, "semivariance": sv, "n_pairs": ct,
        })
    return results


def gaussian_model(h, psill, a, nugget):
    """Gaussian variogram model: nugget + psill*(1 - exp(-3*(h/a)^2))"""
    return nugget + psill * (1.0 - np.exp(-3.0 * (h / a) ** 2))


# ========================================================================
# 2. VARIOGRAM PLOT (publication quality)
# ========================================================================

def plot_variogram(vario_dict, true_params=None, fitted_params=None,
                   engine_name="", scenario_name="", save_path=None):
    """
    Publication-quality variogram plot with empirical points, pair counts,
    and optional fitted/true model curves.

    Parameters
    ----------
    vario_dict : dict from compute_empirical_variogram (omnidirectional)
    true_params : dict  {psill, range_major, nugget}   (ground truth)
    fitted_params : dict {psill, range, nugget}         (engine recovered)
    """
    lags = vario_dict["lags"]
    sv = vario_dict["semivariance"]
    np_ = vario_dict["n_pairs"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # -- upper: semivariance --
    valid = np_ > 0
    ax1.scatter(lags[valid], sv[valid], s=60, c="crimson", edgecolor="k",
                zorder=5, label="Empirical")

    h_fine = np.linspace(0, lags.max() * 1.1, 300)

    if true_params is not None:
        y_true = gaussian_model(h_fine, true_params["psill"],
                                true_params["range_major"],
                                true_params["nugget"])
        ax1.plot(h_fine, y_true, "b-", lw=2, label="True model")

    if fitted_params is not None:
        y_fit = gaussian_model(h_fine, fitted_params.get("psill", 1),
                               fitted_params.get("range", 200),
                               fitted_params.get("nugget", 0))
        ax1.plot(h_fine, y_fit, "g--", lw=2, label=f"Fitted ({engine_name})")

    ax1.set_ylabel("Semivariance", fontsize=12)
    ax1.set_title(f"Empirical Variogram  -  {scenario_name}  [{engine_name}]",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, ls="--", alpha=0.4)

    # lag info text
    ap = vario_dict.get("auto_params", {})
    info_lines = [
        f"Lags: {len(lags)}  |  Lag width: {vario_dict['lag_width']:.1f} m  "
        f"|  Max lag: {vario_dict['max_lag']:.0f} m",
    ]
    if ap:
        info_lines.append(
            f"Median NN: {ap.get('median_nn', 0):.1f} m  "
            f"|  Avg pairs/bin: {ap.get('avg_pairs_per_bin', 0):.0f}"
        )
    info = "\n".join(info_lines)
    ax1.text(0.02, 0.97, info, transform=ax1.transAxes, fontsize=8,
             va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # -- lower: pair counts --
    ax2.bar(lags, np_, width=vario_dict["lag_width"] * 0.8,
            color="steelblue", edgecolor="k", alpha=0.7)
    ax2.set_xlabel("Lag Distance (m)", fontsize=12)
    ax2.set_ylabel("# Pairs", fontsize=12)
    ax2.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_directional_variogram(dir_vario_list, true_params=None,
                               scenario_name="", save_path=None):
    """Plot directional variograms on a single figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for i, vd in enumerate(dir_vario_list):
        valid = vd["n_pairs"] > 0
        ax.plot(vd["lags"][valid], vd["semivariance"][valid], "o-",
                color=cmap(i), ms=5, lw=1.2,
                label=f'{vd["direction"]:.0f} deg')

    if true_params is not None:
        h = np.linspace(0, dir_vario_list[0]["max_lag"], 200)
        y = gaussian_model(h, true_params["psill"],
                           true_params["range_major"], true_params["nugget"])
        ax.plot(h, y, "k--", lw=2, label="True (major axis)")

    ax.set_xlabel("Lag Distance (m)", fontsize=12)
    ax.set_ylabel("Semivariance", fontsize=12)
    ax.set_title(f"Directional Variograms  -  {scenario_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 3. CROSS-VALIDATION (GPR & Kriging)
# ========================================================================

def perform_gpr_kfold_cv(rgpr_model, X, y, n_folds=5, seed=42):
    """
    K-fold spatial cross-validation for RotatedGPR using fixed parameters.

    Returns DataFrame with Observed, Predicted, Uncertainty, Residual, Z_Score.
    """
    from sklearn.cluster import KMeans

    params = rgpr_model.get_kernel_params()
    angle = params["rotation_angle_deg"]
    alpha = params.get("noise_level", rgpr_model.alpha)

    # Fixed kernel from learned theta
    fitted_kernel = clone(rgpr_model.gp_model_.kernel_)

    # Spatial folds
    clusters = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit_predict(X)

    results = []
    for fold in range(n_folds):
        tr = clusters != fold
        te = clusters == fold
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # Center + rotate using model's convention
        if rgpr_model.center_coords:
            center = np.mean(X_tr, axis=0)
            X_tr_c = X_tr - center
            X_te_c = X_te - center
        else:
            X_tr_c, X_te_c = X_tr, X_te

        X_tr_rot = rgpr_model._rotate_coords(X_tr_c, angle)
        X_te_rot = rgpr_model._rotate_coords(X_te_c, angle)

        gp = GaussianProcessRegressor(
            kernel=clone(fitted_kernel), alpha=alpha,
            optimizer=None, normalize_y=True, random_state=seed,
        )
        gp.fit(X_tr_rot, y_tr)
        pred, std = gp.predict(X_te_rot, return_std=True)

        for j in range(len(y_te)):
            res = y_te[j] - pred[j]
            results.append({
                "X": X_te[j, 0], "Y": X_te[j, 1],
                "Observed": float(y_te[j]),
                "Predicted": float(pred[j]),
                "Uncertainty": float(std[j]),
                "Residual": float(res),
                "Z_Score": float(res / (std[j] + 1e-12)),
                "Abs_Error": float(abs(res)),
                "Fold": fold,
            })
    return pd.DataFrame(results)


def perform_kriging_kfold_cv(ak_model, X, y, n_folds=5, seed=42):
    """K-fold spatial CV for AnisotropicKriging."""
    from sklearn.cluster import KMeans
    from pykrige.ok import OrdinaryKriging

    clusters = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit_predict(X)
    bp = dict(ak_model.best_params_)
    model_name = ak_model.best_model_name_

    results = []
    for fold in range(n_folds):
        tr = clusters != fold
        te = clusters == fold
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        try:
            ok = ak_model._get_ok_instance(X_tr, y_tr, bp, model_name)
            pred, var = ok.execute("points", X_te[:, 0], X_te[:, 1])
            std = np.sqrt(np.abs(var))
        except Exception:
            continue

        for j in range(len(y_te)):
            res = y_te[j] - pred[j]
            results.append({
                "X": X_te[j, 0], "Y": X_te[j, 1],
                "Observed": float(y_te[j]),
                "Predicted": float(pred[j]),
                "Uncertainty": float(std[j]),
                "Residual": float(res),
                "Z_Score": float(res / (std[j] + 1e-12)),
                "Abs_Error": float(abs(res)),
                "Fold": fold,
            })
    return pd.DataFrame(results)


# ========================================================================
# 4. CV DASHBOARD PLOT
# ========================================================================

def plot_cv_dashboard(cv_df, engine_name="", scenario_name="",
                      save_path=None):
    """
    3-panel cross-validation dashboard:
      A) Observed vs Predicted scatter + 1:1 line + stats
      B) Spatial map of Z-scores
      C) Consistency bar chart (predicted vs observed with error bars)
    """
    fig = plt.figure(figsize=(14, 9), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # ---- A: Observed vs Predicted ----
    mn = min(cv_df["Observed"].min(), cv_df["Predicted"].min())
    mx = max(cv_df["Observed"].max(), cv_df["Predicted"].max())
    ax1.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="1:1")
    sc = ax1.scatter(cv_df["Observed"], cv_df["Predicted"],
                     c=cv_df["Z_Score"].abs(), cmap="Reds",
                     edgecolor="k", s=30, vmin=0, vmax=3)
    plt.colorbar(sc, ax=ax1, label="|Z-score|", fraction=0.046)

    mae = cv_df["Abs_Error"].mean()
    rmse = np.sqrt((cv_df["Residual"] ** 2).mean())
    rmss = np.sqrt((cv_df["Z_Score"] ** 2).mean())
    corr = cv_df["Observed"].corr(cv_df["Predicted"])
    stats = f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nRMSS={rmss:.3f}\nCorr={corr:.3f}"
    ax1.text(0.05, 0.95, stats, transform=ax1.transAxes, va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax1.set_xlabel("Observed")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Goodness of Fit", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.5)

    # ---- B: Spatial Z-score ----
    sc2 = ax2.scatter(cv_df["X"], cv_df["Y"], c=cv_df["Z_Score"],
                      cmap="RdBu_r", s=30, edgecolor="k", lw=0.5,
                      vmin=-3, vmax=3)
    plt.colorbar(sc2, ax=ax2, label="Z-score", fraction=0.046)
    ax2.set_aspect("equal")
    ax2.set_title("Spatial Z-Score Map", fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5)

    # ---- C: Consistency ----
    df_s = cv_df.sort_values("Observed").reset_index(drop=True)
    idx = np.arange(len(df_s))
    ax3.errorbar(idx, df_s["Predicted"], yerr=1.96 * df_s["Uncertainty"],
                 fmt="none", ecolor="gray", alpha=0.4, capsize=2,
                 label="95% CI")
    ax3.scatter(idx, df_s["Predicted"], c="red", marker="x", s=20,
                label="Predicted", zorder=5)
    ax3.scatter(idx, df_s["Observed"], c="black", marker="o", s=15,
                alpha=0.6, label="Observed", zorder=4)
    ax3.set_xlabel("Point index (sorted by observed)")
    ax3.set_ylabel("Value")
    ax3.set_title("Consistency Check", fontweight="bold")
    ax3.legend(fontsize=8, ncol=3, loc="upper left")
    ax3.grid(True, ls=":", alpha=0.5)

    fig.suptitle(f"CV Dashboard  -  {scenario_name}  [{engine_name}]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ========================================================================
# 5. ANISOTROPY ELLIPSE PLOT
# ========================================================================

def plot_anisotropy_ellipse(params, true_params=None, engine_name="",
                            scenario_name="", save_path=None):
    """Draw correlation ellipse comparing recovered vs true anisotropy."""
    angle = params.get("rotation_angle_deg", 0)
    ls = params.get("length_scale", [1, 1])
    if not hasattr(ls, '__len__'):
        ls = [ls, ls]
    ratio = params.get("anisotropy_ratio", 1)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    # Recovered ellipse
    ell = Ellipse((0, 0), 2 * ls[0], 2 * ls[1], angle=angle,
                  ec="red", fc="salmon", alpha=0.3, lw=2,
                  label=f"Recovered (angle={angle:.1f}, ratio={ratio:.2f})")
    ax.add_patch(ell)

    if true_params is not None:
        ta = true_params.get("angle", 0)
        tr = true_params.get("range_major", 1)
        tmin = true_params.get("range_minor", 1)
        ell2 = Ellipse((0, 0), 2 * tr, 2 * tmin, angle=ta,
                       ec="blue", fc="none", lw=2, ls="--",
                       label=f"True (angle={ta:.1f}, ratio={tr/tmin:.2f})")
        ax.add_patch(ell2)

    lim = max(max(ls), true_params.get("range_major", 1) if true_params else 1) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_title(f"Anisotropy Ellipse  -  {scenario_name}  [{engine_name}]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
