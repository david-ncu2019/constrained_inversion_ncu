from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class SystemConfig:
    """Configuration for the constrained inversion toy problem."""

    grid_size: int = 10  # 10x10 horizontal grid
    n_layers: int = 5  # 5 vertical layers
    well_indices: tuple[int, ...] = (
        11,
        33,
        55,
        77,
        89,
    )  # 0-indexed equivalents of 12, 34, 56, 78, 90

    # Physical units contract
    # ─────────────────────
    # The model vector m represents compaction density in mm/m (compaction per
    # metre of layer thickness).  delta_z converts density to absolute
    # displacement:
    #   - G @ m  sums  f_ij * delta_z  over 0–300 m layers → partial surface
    #     displacement (0–300 m contribution only). InSAR includes deeper signal.
    #   - I_wells selects raw model values (mm/m).  Multiply by delta_z to
    #     obtain mm of layer compaction.
    # When delta_z = 1.0 (toy problem), mm/m and mm are numerically identical.
    # When delta_z = 5.0 (real domain, 5 m layers), the distinction matters.
    #
    # Toy problem default = 1.0 (unit layers); real domain = 5.0 m per layer.
    delta_z: float = 1.0

    # Temporal Parameters
    n_epochs: int = 12  # Number of time steps
    lam_t: float = 1.0  # Temporal regularization trade-off parameter
    lam: float = 1e-2  # Tikhonov trade-off parameter
    ell: float = 3.0  # Horizontal correlation length (grid units)
    ell_z: float = 0.5  # Vertical correlation length (grid units)
    sigma_insar: float = 0.5  # InSAR noise std dev (mm)
    sigma_well: float = 0.1  # Well noise std dev (mm)
    prior_variance: float = 1.0  # sigma^2 for prior covariance

    # Rectangular grid support.
    # When grid_rows > 0 AND grid_cols > 0, n_pixels = grid_rows * grid_cols.
    # Set both to 0 (default) to use the square grid_size × grid_size formula.
    grid_rows: int = 0
    grid_cols: int = 0

    # Sparse InSAR support.
    # When set, build_G_matrix only produces rows for these pixel indices.
    # None (default) means every pixel has an InSAR observation (toy behaviour).
    insar_pixel_indices: tuple[int, ...] | None = None

    # Geographic spatial smoother radius (metres).
    # When > 0, build_L_spatial() connects station pairs within this distance
    # using distance-weighted penalties.  Set to 0 to fall back to the
    # sequential build_L_matrix() (toy/synthetic paths).
    spatial_dist_threshold_m: float = 15000.0

    # Flag indicating whether the dataset is in cumulative displacement space.
    # Set by loader.build_real_dataset(cumulate=True).  Used for diagnostics
    # and to select the correct temporal regularisation interpretation.
    cumulative_space: bool = True

    @property
    def n_pixels(self) -> int:
        if self.grid_rows > 0 and self.grid_cols > 0:
            return self.grid_rows * self.grid_cols
        return self.grid_size * self.grid_size

    @property
    def total_voxels(self) -> int:
        return self.n_pixels * self.n_layers

    @property
    def n_insar_obs(self) -> int:
        if self.insar_pixel_indices is not None:
            return len(self.insar_pixel_indices)
        return self.n_pixels

    @property
    def n_well_obs(self) -> int:
        return len(self.well_indices) * self.n_layers

    @classmethod
    def create_scalable(
        cls, grid_size: int, n_layers: int, num_wells: int, seed: int = 42, **kwargs: float
    ) -> "SystemConfig":
        """Generates a configuration with randomly placed wells for scaled testing."""
        rng = np.random.default_rng(seed)
        n_pixels = grid_size * grid_size
        # Pick 'num_wells' unique pixel indices
        wells = tuple(sorted(rng.choice(n_pixels, size=num_wells, replace=False)))

        # We need to type-ignore kwargs spreading for dataclasses in mypy sometimes,
        # but here we can just unpack carefully or ignore.
        return cls(grid_size=grid_size, n_layers=n_layers, well_indices=wells, **kwargs)  # type: ignore

    @classmethod
    def create_real_domain(
        cls,
        grid_size: int,
        well_indices: tuple[int, ...],
        n_layers: int = 59,
        **kwargs: float,
    ) -> "SystemConfig":
        """Real-domain config for CRFP: 200 m horizontal, 5 m vertical layers."""
        return cls(
            grid_size=grid_size,
            n_layers=n_layers,
            well_indices=well_indices,
            delta_z=5.0,
            sigma_well=kwargs.pop("sigma_well", 1.0),
            sigma_insar=kwargs.pop("sigma_insar", 3.0),
            **kwargs,
        )


@dataclass
class SyntheticDataset:
    """Container for the generated synthetic ground truth and observations."""

    config: SystemConfig
    m_true: npt.NDArray[np.float64]  # shape: (total_voxels,)
    d_insar: npt.NDArray[np.float64]  # shape: (n_insar_obs,)
    w: npt.NDArray[np.float64]  # shape: (n_well_obs,)
