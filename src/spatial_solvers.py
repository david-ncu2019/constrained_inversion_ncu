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

    def _constrained_optimizer(self, obj_func, initial_theta, bounds):
        """Custom optimizer wrapper that enforces anisotropy ratio constraints using SLSQP."""
        from scipy.optimize import minimize
        
        # Find indices of length_scale parameters
        ls_indices = None
        idx = 0
        for hyper in self.kernel.hyperparameters:
            if 'length_scale' in hyper.name:
                if hyper.n_elements == 2:
                    ls_indices = [idx, idx + 1]
                    break
            idx += hyper.n_elements
        
        # Fallback
        if ls_indices is None:
            ls_indices = [1, 2]
        
        # Define constraint: |log(L1) - log(L2)| <= log(max_ratio)
        log_ratio_limit = np.log(self.max_anisotropy)
        
        def ratio_constraint(theta):
            return log_ratio_limit - np.abs(theta[ls_indices[0]] - theta[ls_indices[1]])
        
        constraint = {'type': 'ineq', 'fun': ratio_constraint}
        
        result = minimize(
            obj_func,
            initial_theta,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=[constraint],
            options={'ftol': 1e-9, 'maxiter': 100}
        )
        
        return result.x, result.fun

    def _angle_objective(self, angle: float) -> float:
        """Negative log marginal likelihood for a given rotation angle."""
        X_rot = self._rotate_coords(self.X_train_centered_, angle)
        
        optimizer_func = self._constrained_optimizer if self.max_anisotropy is not None else self.optimizer
        
        gp = GaussianProcessRegressor(
            kernel=clone(self.kernel),
            alpha=self.alpha,
            optimizer=optimizer_func,
            n_restarts_optimizer=0,
            normalize_y=True,
            random_state=self.random_state
        )
        gp.fit(X_rot, self.y_train_)
        
        lml = gp.log_marginal_likelihood()
        self.angle_search_history_.append((angle, -lml))
        
        return -lml

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RotatedGPR':
        """Fit the rotated GPR model by optimizing the rotation angle and kernel hyperparameters."""
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
        
        if self.angle_search_method == "bounded":
            result = minimize_scalar(
                self._angle_objective,
                bounds=self.angle_bounds,
                method='bounded',
                options={'xatol': self.angle_precision}
            )
            self.best_angle_deg_ = result.x
        elif self.angle_search_method == "global":
            result = differential_evolution(
                lambda x: self._angle_objective(x[0]),
                bounds=[self.angle_bounds],
                seed=self.random_state,
                atol=self.angle_precision,
                tol=0.01
            )
            self.best_angle_deg_ = result.x[0]
        else:
            raise ValueError(f"Unknown angle_search_method: {self.angle_search_method}")
        
        # Fit final model
        X_final = self._rotate_coords(self.X_train_centered_, self.best_angle_deg_)
        optimizer_func = self._constrained_optimizer if self.max_anisotropy is not None else self.optimizer
        
        self.gp_model_ = GaussianProcessRegressor(
            kernel=clone(self.kernel),
            alpha=self.alpha,
            optimizer=optimizer_func,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=self.random_state
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
        noise = 0.0
        
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
