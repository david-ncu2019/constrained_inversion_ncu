# src/spatial_solvers.py
"""
Rotated Anisotropic Gaussian Process Regression with automatic angle optimization.
Mathematical enhancements for geological/spatial applications.
"""

from typing import Optional, Tuple, Any, List, Dict

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
import optuna
import matplotlib.pyplot as plt
import os
import warnings


class RotatedGPR(BaseEstimator, RegressorMixin):
    """
    Gaussian Process Regressor with automatic anisotropy rotation optimization.
    
    Learns optimal rotation angle for anisotropic spatial correlation structure.
    Uses two-level optimization: outer loop for angle, inner for kernel hyperparameters.
    
    Parameters
    ----------
    kernel : sklearn kernel object
        Base kernel (should support anisotropic length scales)
    alpha : float, default=1e-10
        Regularization parameter
    optimizer : str, default="fmin_l_bfgs_b"
        Internal optimizer for hyperparameters
    n_restarts_optimizer : int, default=10
        Number of restarts for final model fitting
    angle_search_method : str, default="bounded"
        Method for angle optimization: "bounded" (fast) or "global" (thorough)
    angle_precision : float, default=0.5
        Angular precision in degrees for optimization
    center_coords : bool, default=True
        Whether to center coordinates before rotation
    max_anisotropy : float, optional
        Maximum allowed anisotropy ratio (primary_range / secondary_range).
        If specified, constrains optimization to prevent overly elongated ellipses.
    random_state : int, optional
        Random seed for reproducibility
    angle_bounds : tuple, default=(0, 180)
        Bounds for rotation angle search
    """
    
    def __init__(
        self,
        kernel,
        alpha: float = 1e-10,
        optimizer: str = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 10,
        angle_search_method: str = "bounded",
        angle_precision: float = 0.5,
        center_coords: bool = True,
        max_anisotropy: Optional[float] = None,
        random_state: Optional[int] = None,
        angle_bounds: Tuple[float, float] = (0, 180)
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.angle_search_method = angle_search_method
        self.angle_precision = angle_precision
        self.center_coords = center_coords
        self.max_anisotropy = max_anisotropy
        self.random_state = random_state
        self.angle_bounds = angle_bounds
        
        # Fitted attributes
        self.best_angle_deg_ = None
        self.gp_model_ = None
        self.X_center_ = None
        self.log_marginal_likelihood_ = None
        self.angle_search_history_ = []

    def _center_coordinates(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Center coordinates around their mean."""
        center = np.mean(X, axis=0)
        return X - center, center

    def _rotate_coords(self, X: np.ndarray, angle_deg: float) -> np.ndarray:
        """Apply 2D rotation transformation (counter-clockwise)."""
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ])
        return X @ rotation_matrix

    def _eval_lml(self, angle, theta, alpha):
        """Evaluate negative log-marginal-likelihood for a parameter vector."""
        test_kernel = clone(self.kernel)
        if len(test_kernel.bounds) != len(theta):
            return float('inf')
        test_kernel.theta = theta
        X_rot = self._rotate_coords(self.X_train_centered_, angle)
        gp = GaussianProcessRegressor(
            kernel=test_kernel, alpha=alpha,
            optimizer=None, normalize_y=True,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_rot, self.y_train_)
                return -gp.log_marginal_likelihood()
        except Exception:
            return float('inf')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RotatedGPR':
        """Fit using joint Optuna + L-BFGS-B refinement over angle, kernel, and noise."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.float64)

        if self.X_train_.shape[1] != 2:
            raise ValueError(f"X must have exactly 2 columns, got {self.X_train_.shape[1]}")

        if self.center_coords:
            self.X_train_centered_, self.X_center_ = self._center_coordinates(self.X_train_)
        else:
            self.X_train_centered_ = self.X_train_
            self.X_center_ = np.zeros(2)

        self.angle_search_history_ = []

        # Kernel parameter bounds (log-space)
        k_bounds = self.kernel.bounds  # shape (n_params, 2)
        n_kp = len(k_bounds)
        max_ratio_log = np.log(self.max_anisotropy) if self.max_anisotropy and self.max_anisotropy > 1.0 else 0.0

        # ---- Phase 1: Optuna global search --------------------------------
        def objective(trial):
            angle = trial.suggest_float('angle', self.angle_bounds[0], self.angle_bounds[1])
            c_val = trial.suggest_float('theta_0', k_bounds[0][0], k_bounds[0][1])
            l_maj = trial.suggest_float('l_maj', k_bounds[1][0], k_bounds[1][1])
            ratio_log = trial.suggest_float('ratio_log', 0.0, max_ratio_log) if max_ratio_log > 0 else 0.0
            log_alpha = trial.suggest_float('log_alpha', -14, 0)  # log10 scale

            theta = np.array([c_val, l_maj, l_maj - ratio_log])
            alpha = 10.0 ** log_alpha
            nlml = self._eval_lml(angle, theta, alpha)
            if np.isfinite(nlml):
                self.angle_search_history_.append((angle, nlml))
            return nlml

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=200)
        self.study_ = study

        bp = study.best_params
        best_angle = bp['angle']
        best_c     = bp['theta_0']
        best_lmaj  = bp['l_maj']
        best_ratio = bp.get('ratio_log', 0.0)
        best_la    = bp['log_alpha']

        # ---- Phase 2: L-BFGS-B local refinement ---------------------------
        from scipy.optimize import minimize as sp_minimize

        def _pack(angle, c, lmaj, ratio, la):
            return np.array([angle, c, lmaj, ratio, la])

        def _obj_vec(vec):
            a, c, lm, r, la = vec
            theta = np.array([c, lm, lm - r])
            return self._eval_lml(a, theta, 10.0 ** la)

        x0 = _pack(best_angle, best_c, best_lmaj, best_ratio, best_la)
        bounds_refine = [
            self.angle_bounds,
            (k_bounds[0][0], k_bounds[0][1]),
            (k_bounds[1][0], k_bounds[1][1]),
            (0.0, max_ratio_log if max_ratio_log > 0 else 0.01),
            (-14, 0),
        ]
        try:
            res = sp_minimize(_obj_vec, x0, method='L-BFGS-B',
                              bounds=bounds_refine,
                              options={'ftol': 1e-12, 'maxiter': 200})
            if res.fun < study.best_value:
                best_angle, best_c, best_lmaj, best_ratio, best_la = res.x
        except Exception:
            pass  # keep Optuna result

        # ---- Store best params --------------------------------------------
        self.best_angle_deg_ = float(best_angle)
        self.best_alpha_     = float(10.0 ** best_la)

        best_theta = np.array([best_c, best_lmaj, best_lmaj - best_ratio])
        best_kernel = clone(self.kernel)
        best_kernel.theta = best_theta

        X_final = self._rotate_coords(self.X_train_centered_, self.best_angle_deg_)
        self.gp_model_ = GaussianProcessRegressor(
            kernel=best_kernel, alpha=self.best_alpha_,
            optimizer=None, normalize_y=True,
            random_state=self.random_state,
        )
        self.gp_model_.fit(X_final, self.y_train_)
        self.log_marginal_likelihood_ = self.gp_model_.log_marginal_likelihood()

        return self

    def predict(self, X: np.ndarray, return_std: bool = False, return_cov: bool = False):
        """Predict using the fitted model in the rotated space."""
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.X_center_ if self.center_coords else X
        X_rot = self._rotate_coords(X_centered, self.best_angle_deg_)
        return self.gp_model_.predict(X_rot, return_std=return_std, return_cov=return_cov)

    @property
    def kernel_(self):
        return self.gp_model_.kernel_

    def get_kernel_params(self) -> Dict[str, Any]:
        """Extract learned kernel parameters."""
        kernel_params = self.kernel_.get_params()
        
        constant = 1.0
        length_scale = [1.0, 1.0]
        noise = getattr(self, 'best_alpha_', self.alpha)
        
        for key, val in kernel_params.items():
            if "constant_value" in key and "bounds" not in key:
                constant = float(val)
            elif "length_scale" in key and "bounds" not in key:
                length_scale = val if hasattr(val, '__len__') else [val, val]
            elif "noise_level" in key and "bounds" not in key:
                noise = float(val)
        
        length_scale = np.asarray(length_scale, dtype=np.float64)
        anisotropy_ratio = np.max(length_scale) / np.min(length_scale)
        
        return {
            "rotation_angle_deg": float(self.best_angle_deg_),
            "constant_value": constant,
            "length_scale": length_scale.tolist(),
            "noise_level": noise,
            "anisotropy_ratio": float(anisotropy_ratio),
            "log_marginal_likelihood": float(self.log_marginal_likelihood_)
        }


def validate_anisotropy_assumptions(X: np.ndarray, y: np.ndarray, n_directions: int = 8) -> Dict[str, Any]:
    """Validate if data exhibits directional anisotropy by checking variance ratio across directions."""
    angles = np.linspace(0, 180, n_directions, endpoint=False)
    directional_variances = []
    
    for angle in angles:
        theta = np.radians(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        X_rot = X @ R.T
        primary_coord = X_rot[:, 0]
        
        n_bins = min(10, len(y) // 3)
        if n_bins < 2:
            directional_variances.append(np.var(y))
            continue
            
        bins = np.linspace(primary_coord.min(), primary_coord.max(), n_bins)
        bin_indices = np.digitize(primary_coord, bins)
        
        bin_vars = [np.var(y[bin_indices == i]) for i in range(1, n_bins) if (bin_indices == i).sum() > 1]
        
        directional_variances.append(np.mean(bin_vars) if bin_vars else 0)
    
    directional_variances = np.array(directional_variances)
    var_ratio = np.max(directional_variances) / (np.min(directional_variances) + 1e-10)
    
    return {
        "angles": angles.tolist(),
        "directional_variances": directional_variances.tolist(),
        "variance_ratio": float(var_ratio),
        "is_anisotropic": var_ratio > 1.5
    }


# --- Custom Variogram Models for PyKrige ---

def stable_variogram_model(params, dists):
    """
    Stable variogram model: psill * (1 - exp(-(h/r)^alpha)) + nugget
    params: [psill, range, nugget, alpha]
    """
    psill, r, nugget, alpha = params
    return psill * (1.0 - np.exp(-(dists / r) ** alpha)) + nugget


def circular_variogram_model(params, dists):
    """
    Circular variogram model.
    params: [psill, range, nugget]
    """
    psill, r, nugget = params
    
    # Handle dists > r
    mask = dists <= r
    val = np.full_like(dists, psill + nugget)
    
    h_r = dists[mask] / r
    term = (2.0 / np.pi) * (np.arccos(h_r) - h_r * np.sqrt(1.0 - h_r**2))
    val[mask] = psill * (1.0 - term) + nugget
    return val


def rational_quadratic_variogram_model(params, dists):
    """
    Rational Quadratic variogram model: psill * (1 - (1 + h^2 / (2*alpha*r^2))^-alpha) + nugget
    params: [psill, range, nugget, alpha]
    """
    psill, r, nugget, alpha = params
    return psill * (1.0 - (1.0 + dists**2 / (2.0 * alpha * r**2)) ** (-alpha)) + nugget


class AnisotropicKriging(BaseEstimator, RegressorMixin):
    """
    Ordinary Kriging with Optuna-based anisotropic parameter optimization.
    Supports all native PyKrige models plus custom Stable, Circular, and Rational Quadratic models.
    """
    
    NATIVE_MODELS = ['linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect']
    CUSTOM_MODELS = {
        'stable': stable_variogram_model,
        'circular': circular_variogram_model,
        'rational-quadratic': rational_quadratic_variogram_model
    }

    def __init__(
        self,
        n_trials: int = 50,
        n_splits: int = 5,
        verbose: bool = False,
        random_state: Optional[int] = None,
        max_anisotropy: float = 3.0
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.verbose = verbose
        self.random_state = random_state
        self.max_anisotropy = max_anisotropy
        
        # Fitted attributes
        self.model_ = None
        self.best_params_ = None
        self.best_model_name_ = None
        self.study_ = None

    def _get_ok_instance(self, X, y, params, model_name):
        from pykrige.ok import OrdinaryKriging
        
        # Extract common params
        psill = params['psill']
        v_range = params['range']
        nugget = params['nugget']
        angle = params['angle']
        scaling = params['scaling']
        
        kwargs = {
            'variogram_model': model_name if model_name in self.NATIVE_MODELS else 'custom',
            'anisotropy_angle': angle,
            'anisotropy_scaling': scaling,
            'verbose': False,
            'enable_plotting': False
        }
        
        if model_name in self.NATIVE_MODELS:
            kwargs['variogram_parameters'] = [psill, v_range, nugget]
        else:
            kwargs['variogram_function'] = self.CUSTOM_MODELS[model_name]
            v_params = [psill, v_range, nugget]
            if model_name in ['stable', 'rational-quadratic']:
                v_params.append(params['alpha'])
            kwargs['variogram_parameters'] = v_params
            
        return OrdinaryKriging(X[:, 0], X[:, 1], y, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AnisotropicKriging':
        """Optimize Kriging parameters using Optuna and fit the final model."""
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Pre-compute spatial fold assignments (expensive KMeans done once)
        clusters = KMeans(
            n_clusters=self.n_splits, random_state=self.random_state, n_init=10
        ).fit_predict(X)
        
        # Tighter bounds derived from data
        data_var   = float(np.var(y))
        max_dist   = float(np.sqrt((X[:, 0].max() - X[:, 0].min())**2 +
                                   (X[:, 1].max() - X[:, 1].min())**2))
        
        def objective(trial):
            model_name = trial.suggest_categorical(
                'model', self.NATIVE_MODELS + list(self.CUSTOM_MODELS.keys())
            )
            
            psill   = trial.suggest_float('psill',   data_var * 0.05, data_var * 3.0)
            v_range = trial.suggest_float('range',   max_dist * 0.02, max_dist * 1.5)
            nugget  = trial.suggest_float('nugget',  0.0, data_var * 0.8)
            angle   = trial.suggest_float('angle',   0.0, 180.0)
            scaling = trial.suggest_float('scaling', 1.0, self.max_anisotropy)
            
            params = {
                'psill': psill, 'range': v_range, 'nugget': nugget,
                'angle': angle, 'scaling': scaling
            }
            
            if model_name in ['stable', 'rational-quadratic']:
                params['alpha'] = trial.suggest_float('alpha', 0.1, 2.0)
            
            scores = []
            for fold in range(self.n_splits):
                train_idx = np.where(clusters != fold)[0]
                val_idx   = np.where(clusters == fold)[0]
                
                if len(train_idx) < 5 or len(val_idx) < 1:
                    continue
                
                try:
                    ok = self._get_ok_instance(
                        X[train_idx], y[train_idx], params, model_name
                    )
                    y_pred, _ = ok.execute('points', X[val_idx, 0], X[val_idx, 1])
                    mse = float(np.mean((y[val_idx] - y_pred) ** 2))
                    scores.append(mse)
                except Exception:
                    return float('inf')
            
            return np.mean(scores) if scores else float('inf')

        self.study_ = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        self.study_.optimize(objective, n_trials=self.n_trials)
        
        self.best_params_ = self.study_.best_params
        self.best_model_name_ = self.best_params_.pop('model')
        
        # Fit final model
        self.model_ = self._get_ok_instance(X, y, self.best_params_, self.best_model_name_)
        
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Predict using the optimized Kriging model."""
        X = np.asarray(X)
        y_pred, y_var = self.model_.execute('points', X[:, 0], X[:, 1])
        if return_std:
            return y_pred, np.sqrt(np.abs(y_var))
        return y_pred

    def save_diagnostics(self, output_path: str, layer_idx: int):
        """Save variogram plot and parameters."""
        os.makedirs(output_path, exist_ok=True)

        # Plot variogram
        plt.figure(figsize=(8, 5))
        
        try:
            # Manually extract and plot to avoid blocking windows from PyKrige
            lags = self.model_.lags
            semiv = self.model_.semivariance
            plt.plot(lags, semiv, 'ro', label='Empirical Semivariogram')
            
            if callable(self.model_.variogram_function):
                x = np.linspace(0, np.max(lags), 100)
                y = self.model_.variogram_function(self.model_.variogram_model_parameters, x)
                plt.plot(x, y, 'b-', label=f'{self.best_model_name_} Model')
        except AttributeError:
            # Fallback if internal names change
            pass

        plt.title(f"Layer {layer_idx} Variogram - Model: {self.best_model_name_}")
        plt.xlabel('Distance')
        plt.ylabel('Semivariance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"layer_{layer_idx}_variogram.png"))
        plt.close()
        
        # Parameters
        with open(os.path.join(output_path, f"layer_{layer_idx}_params.txt"), 'w') as f:
            f.write(f"Best Model: {self.best_model_name_}\n")
            for k, v in self.best_params_.items():
                f.write(f"{k}: {v}\n")

    def get_kernel_params(self) -> Dict[str, Any]:
        """Adhere to a similar structure as RotatedGPR for reporting."""
        params = {
            "model_type": "Kriging",
            "best_model": self.best_model_name_,
            "rotation_angle_deg": float(self.best_params_.get('angle', 0)),
            "anisotropy_ratio": float(self.best_params_.get('scaling', 1.0)),
            "psill": float(self.best_params_.get('psill', 0)),
            "range": float(self.best_params_.get('range', 0)),
            "nugget": float(self.best_params_.get('nugget', 0)),
        }
        if 'alpha' in self.best_params_:
            params['alpha'] = float(self.best_params_['alpha'])
        return params
