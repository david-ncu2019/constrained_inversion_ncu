# modification_suggestions.md

**Purpose**: This document is a complete audit of every hardcoded assumption, magic
number, fixed dimension, fixed variable name, and fixed file path found across the
CRFP subsidence inversion pipeline. It is written so a developer with no prior
knowledge of this project can implement every change line-by-line without asking
questions.

**Audience**: Junior developer / assistant.

**Working directory of all scripts**: `D:\110_PROJECT_002\constrained_inversion_scripts\`

---

## What "Flexibility" Means in This Pipeline

The pipeline currently assumes a specific dataset: 31–39 monitoring wells, 84 monthly
epochs (Jan 2015 – Dec 2021), 30 depth layers (10 m–290 m, 10 m spacing), and a
153×117 InSAR grid. Any dataset that differs in any of these dimensions will silently
produce wrong results or crash.

A flexible pipeline auto-detects everything from its input files:

| Dimension | Current state | Goal |
|---|---|---|
| Number of stations | Assumed from CSV count | Read from coordinate CSV at runtime |
| Number of depth layers | Discovered by `find_valid_mlcw_depths` (good) | Keep as-is; propagate everywhere |
| Number of time epochs | Discovered from first CSV (good) | Fold boundaries must scale with it |
| Fold boundaries (months) | Hardcoded: 47, 59, 71, 72, 82 | Computed as fractions of total epochs |
| Temporal RMSE assignment | Hardcoded: folds cover 0–59, 60–71, 72+ | Computed from fold definitions |
| Grid dimensions | 153×117 hardcoded in comments and docs | Always read from NetCDF at runtime |
| GP kernel bounds | Fixed: length_scale 1–50 km, noise 1e-5–0.1 | Configurable in `pipeline_config.ini` |
| Variable names in NetCDF | Hardcoded: `"displacement"`, `"Depth"`, `"X"`, `"Y"`, `"Time"` | Configurable |
| CSV column prefix | Hardcoded: `"Month_"` | Configurable |
| CSV filename pattern | Hardcoded: `"{name}_insar_mlcw.csv"` | Configurable |
| Depth column name | Hardcoded: `"Depth"` | Configurable |
| InSAR row index | Hardcoded: depth == 0 | Configurable |
| Windows encoding | `subprocess` default encoding causes Unicode errors | Force UTF-8 (already partially done) |
| Hyperparameter search range | lam_t floor = 0.1, no values < 0.1 | Lower bound configurable |

---

## Priority Labels

- **Critical** — Will crash or silently produce wrong results when data dimensions change.
- **Important** — Will work on the original data but breaks on any other dataset.
- **Nice-to-have** — No correctness risk; improves usability or maintainability.

---

## File: `cv_temporal_forward.py`

### Issue 1 — Hardcoded fold boundaries (Known Issue #3, Critical)

**Lines 63–67**:
```python
FOLDS = [
    {"fold": 1, "train_month_end": 47,  "val_start": 48, "val_end": 59},
    {"fold": 2, "train_month_end": 59,  "val_start": 60, "val_end": 71},
    {"fold": 3, "train_month_end": 71,  "val_start": 72, "val_end": 82},
]
```

**Why it is a problem**: These numbers assume exactly 84 monthly epochs (Jan 2015 –
Dec 2021). The synthetic dataset has fewer epochs. Folds 2 and 3 silently return NaN
because `val_start` (60, 72) exceed the synthetic dataset length. The pipeline reports
zero NaN warnings and continues, giving meaningless CV metrics.

**Fix**: Replace the module-level constant `FOLDS` with a function that computes fold
boundaries dynamically from the actual number of epochs. Add a call at the top of
`main()` and in `run_training_fold()` (since `run_training_fold` is imported and called
from `tune_hyperparams.py` without going through `main()`).

Replace lines 63–67 with:

```python
def compute_folds(n_epochs: int) -> list[dict]:
    """
    Compute forward-chaining fold boundaries from the total number of epochs.

    Uses a three-fold scheme: 60% / 20% / 20% split (approximate), with a
    minimum of 6 epochs per fold.  The validation windows are non-overlapping
    and contiguous.

    Parameters
    ----------
    n_epochs : int
        Total number of monthly epochs in the dataset.

    Returns
    -------
    list of dict, each with keys:
        fold (int), train_month_end (int), val_start (int), val_end (int)
    """
    if n_epochs < 12:
        raise ValueError(
            f"Dataset has only {n_epochs} epochs. Need at least 12 for 3-fold CV."
        )
    # Each validation window is ~13% of total epochs, minimum 6 months.
    val_len = max(6, n_epochs // 8)
    # Fold 3: last val_len epochs
    f3_end = n_epochs - 1
    f3_start = f3_end - val_len + 1
    # Fold 2: immediately before fold 3
    f2_end = f3_start - 1
    f2_start = f2_end - val_len + 1
    # Fold 1: immediately before fold 2
    f1_end = f2_start - 1
    f1_start = f1_end - val_len + 1
    if f1_start < 1:
        raise ValueError(
            f"Dataset has only {n_epochs} epochs. Cannot build 3 non-overlapping "
            "folds with sufficient training data. Reduce val_len or add more epochs."
        )
    return [
        {"fold": 1, "train_month_end": f1_start - 1, "val_start": f1_start, "val_end": f1_end},
        {"fold": 2, "train_month_end": f2_start - 1, "val_start": f2_start, "val_end": f2_end},
        {"fold": 3, "train_month_end": f3_start - 1, "val_start": f3_start, "val_end": f3_end},
    ]


# Module-level fallback for backward compatibility (84-epoch real dataset).
# Replaced at runtime inside main() by compute_folds(n_epochs).
FOLDS = compute_folds(84)
```

Then in `main()`, add the following immediately after line 446
(`plots_dir = args.output_dir / "plots"`):

```python
    # Auto-detect number of epochs and recompute fold boundaries.
    _sample_csv = next((args.data_dir / "CSV_files").glob("*_insar_mlcw.csv"))
    _sample_df = pd.read_csv(_sample_csv)
    _n_epochs = len([c for c in _sample_df.columns if c.startswith("Month_")])
    folds = compute_folds(_n_epochs)
    print(f"Detected {_n_epochs} epochs. Fold boundaries: {folds}")
```

Then replace the loop variable `for fold_def in FOLDS:` (line 457) with:

```python
    for fold_def in folds:
```

Also replace the `max_val_len` calculation (line 452):

```python
    max_val_len = max(f["val_end"] - f["val_start"] + 1 for f in FOLDS)
```

with:

```python
    max_val_len = max(f["val_end"] - f["val_start"] + 1 for f in folds)
```

---

### Issue 2 — Hardcoded `"Month_"` prefix (Important)

**Lines 178–179**:
```python
    val_month_labels = [
        f"Month_{k + 1:03d}" for k in range(val_month_start, val_month_end + 1)
    ]
```

**Why it is a problem**: The column naming convention `Month_001`, `Month_002`, ...
is specific to this dataset. A different dataset might use `month_0`, `time_001`, etc.
If the prefix or numbering offset changes, all validation labels will be wrong, and
every CSV lookup at line 196–210 (`if mc in df.columns`) will silently return NaN for
every entry.

**Note**: The numbering convention in the CSV files uses 1-based indices
(`Month_001` = epoch 0 in 0-based Python arrays). This offset is already captured
correctly here but is not documented. The fix below makes the prefix configurable and
adds the offset as a named constant.

Replace the `load_validation_data` function signature (line 149) with:

```python
def load_validation_data(
    data_dir: Path,
    station_names: list[str],
    valid_depths_m: list[int],
    val_month_start: int,
    val_month_end: int,
    month_col_prefix: str = "Month_",
    month_col_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
```

Replace lines 178–181 with:

```python
    val_month_labels = [
        f"{month_col_prefix}{k + month_col_offset:03d}"
        for k in range(val_month_start, val_month_end + 1)
    ]
```

All callers of `load_validation_data` in `cv_temporal_forward.py` and
`tune_hyperparams.py` already pass only positional arguments that match the new
defaults, so they require no changes.

---

### Issue 3 — Hardcoded CSV filename pattern (Important)

**Line 189**:
```python
        csv_path = csv_dir / f"{name}_insar_mlcw.csv"
```

**Why it is a problem**: The filename pattern `{name}_insar_mlcw.csv` is assumed in
at least five places across the codebase (`loader.py`, `cv_temporal_forward.py`,
`tune_hyperparams.py` indirectly, `cv_spatial_gp.py` indirectly via loader). If the
user's files are named differently (e.g., `{name}.csv` or `station_{name}_data.csv`),
all of these paths silently fail (`csv_path.exists()` returns False) and the station is
treated as having no data.

**Fix**: Add a `csv_name_pattern` parameter to `load_validation_data`.

Add to function signature:

```python
    csv_name_pattern: str = "{name}_insar_mlcw.csv",
```

Replace line 189 with:

```python
        csv_path = csv_dir / csv_name_pattern.format(name=name)
```

---

### Issue 4 — `"Depth"` and `0` hardcoded as InSAR row index (Important)

**Lines 192–200**:
```python
        df = pd.read_csv(csv_path).set_index("Depth")

        # InSAR surface (depth = 0)
        if 0 in df.index:
            for t_idx, mc in enumerate(val_month_labels):
                if mc in df.columns:
                    val = df.loc[0, mc]
```

**Why it is a problem**: The column named `"Depth"` is used as the index, and the
special value `0` identifies the InSAR surface row. Any dataset with a different
index column name or a different sentinel value for the surface observation will
silently skip all InSAR data.

**Fix**: Add parameters `depth_col: str = "Depth"` and `insar_depth_val: int = 0` to
`load_validation_data`. Then replace the hardcoded strings:

```python
        df = pd.read_csv(csv_path).set_index(depth_col)

        # InSAR surface row
        if insar_depth_val in df.index:
            for t_idx, mc in enumerate(val_month_labels):
                if mc in df.columns:
                    val = df.loc[insar_depth_val, mc]
```

---

## File: `tune_hyperparams.py`

### Issue 5 — Hardcoded fallback fold boundaries (Known Issue #3, Critical)

**Lines 53–55**:
```python
_FOLD3_TRAIN_END = 71
_FOLD3_VAL_START = 72
_FOLD3_VAL_END = 82
```

**Why it is a problem**: These three module-level globals are fallback values for
fold 3 of the 84-epoch dataset. `update_fold_split()` is supposed to replace them
but only runs when `main()` runs. When `evaluate_one_config` is imported and called
directly from `run_pipeline.py` (which calls `tune_hyperparams.py` via subprocess), the
update does run. However, when a developer imports only `evaluate_one_config` in a
notebook or test without calling `main()`, the stale values are used.

**Fix**: Remove the module-level globals. Make `evaluate_one_config` accept a
`fold3_def` parameter computed by `compute_folds` (imported from
`cv_temporal_forward`).

Replace lines 44–68 with:

```python
from cv_temporal_forward import (
    compute_folds,
    compute_fold_metrics,
    load_validation_data,
    predict_validation_compaction,
    prepare_validation_arrays,
    run_training_fold,
)


def _get_fold3(data_dir: Path) -> dict:
    """
    Detect the number of epochs from the first station CSV and return
    the fold-3 definition dict produced by compute_folds().

    Parameters
    ----------
    data_dir : Path

    Returns
    -------
    dict with keys: fold, train_month_end, val_start, val_end
    """
    csv_dir = data_dir / "CSV_files"
    sample_csv = next(csv_dir.glob("*_insar_mlcw.csv"))
    df = pd.read_csv(sample_csv)
    month_cols = [c for c in df.columns if c.startswith("Month_")]
    n_epochs = len(month_cols)
    return compute_folds(n_epochs)[2]   # index 2 = fold 3
```

Then update `evaluate_one_config` signature:

```python
def evaluate_one_config(
    data_dir: Path,
    lam: float,
    lam_t: float,
    sigma_insar: float = 3.0,
    sigma_well: float = 1.0,
    fold3_def: dict | None = None,
) -> dict:
```

At the top of `evaluate_one_config`, replace references to module-level globals:

```python
    if fold3_def is None:
        fold3_def = _get_fold3(data_dir)
    train_end = fold3_def["train_month_end"]
    val_start = fold3_def["val_start"]
    val_end   = fold3_def["val_end"]
```

Then replace lines 104, 119, 127 (the three uses of the module-level variables):

```python
    depth_weights, meta = run_training_fold(
        data_dir=data_dir,
        train_month_end=train_end,
        ...
    )
    ...
    insar_val_inc, mlcw_val_inc, _ = load_validation_data(
        ...
        val_month_start=val_start,
        val_month_end=val_end,
    )
    ...
    per_layer_df, _ = compute_fold_metrics(
        pred, mlcw_val_cum, valid_depths_m, station_names, fold_id=fold3_def["fold"]
    )
```

In `main()`, replace the `update_fold_split` call (line 240) with:

```python
    fold3_def = _get_fold3(args.data_dir)
    print(
        f"Fold-3 definition: train 0–{fold3_def['train_month_end']}, "
        f"validate {fold3_def['val_start']}–{fold3_def['val_end']}"
    )
```

Update the print statement (line 246) to use `fold3_def`:

```python
    print(
        f"Hyperparameter grid search — fold 3 only "
        f"(train 0–{fold3_def['train_month_end']}, "
        f"validate {fold3_def['val_start']}–{fold3_def['val_end']})"
    )
```

Pass `fold3_def` through the evaluation loop (lines 255–267):

```python
    for k, (lam, lam_t) in enumerate(product(lam_candidates, lam_t_candidates), start=1):
        ...
        result = evaluate_one_config(
            data_dir=args.data_dir,
            lam=lam,
            lam_t=lam_t,
            sigma_insar=args.sigma_insar,
            sigma_well=args.sigma_well,
            fold3_def=fold3_def,
        )
```

---

### Issue 6 — Default `lam_t` candidates miss values below 0.1 (Known Issue #5, Important)

**Line 71**:
```python
DEFAULT_LAM_T_CANDIDATES = [0.1, 0.3, 1.0, 3.0]
```

**Line 475 in `run_pipeline.py`**:
```python
lam_t_cands = cfg.get("Tuning", "lam_t_candidates", fallback="0.1,0.3,1.0,3.0")
```

**Why it is a problem**: For synthetic datasets (few stations, short time series), the
optimal `lam_t` is often below 0.1. The grid search never explores this region and
reports the best result as 0.1, which may be significantly suboptimal.

**Fix in `tune_hyperparams.py`** — Change line 71:

```python
DEFAULT_LAM_T_CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
```

**Fix in `pipeline_config.ini`** — Change line 34:

```ini
lam_t_candidates = 0.01,0.03,0.1,0.3,1.0,3.0
```

**Fix in `synthetic_config.ini`** — Change the same line (line 34 of that file):

```ini
lam_t_candidates = 0.01,0.03,0.1,0.3,1.0,3.0
```

---

### Issue 7 — `update_fold_split` is a global side-effect mutation (Nice-to-have)

**Lines 57–68**: The function `update_fold_split` mutates three module-level globals
(`_FOLD3_TRAIN_END`, `_FOLD3_VAL_START`, `_FOLD3_VAL_END`) via `global` statements.
This pattern is fragile: if the function is called concurrently (e.g., during parallel
hyperparameter search) or from a different thread, the globals can be in an
inconsistent state.

This issue is fully resolved by Issue 5's fix above (the globals are eliminated). No
additional fix is needed if Issue 5 is applied.

---

## File: `stage2_gp_interpolation.py`

### Issue 8 — Hardcoded NetCDF variable name `"displacement"` (Critical)

**Line 149**:
```python
    disp = ds["displacement"].sel(Depth=0).values  # (n_time, n_y, n_x), float32
```

**Why it is a problem**: The variable name `"displacement"` and the dimension
selector `Depth=0` are specific to the real CRFP NetCDF file
(`grid_pnt_CRFP_500m_vert_IDW_v1.nc`). The synthetic grid file
(`grid_pnt_datacube_500m.nc`) has the same variable name by coincidence. Any
third-party NetCDF with a different variable name (e.g., `"subsidence"`,
`"los_displacement"`, `"vert_disp"`) will raise `KeyError` on line 149.

**Fix**: Add a `--displacement-var` argument to `parse_args()` and pass it to
`load_grid_insar`.

Add to `parse_args()` (after line 85):

```python
    p.add_argument(
        "--displacement-var", default="displacement", type=str,
        help="Name of the displacement variable in the grid NetCDF (default: 'displacement').",
    )
    p.add_argument(
        "--insar-depth-dim", default="Depth", type=str,
        help="Name of the depth dimension in the grid NetCDF (default: 'Depth').",
    )
    p.add_argument(
        "--insar-surface-depth", default=0, type=int,
        help="Value of the depth dimension that selects the InSAR surface slice "
             "(default: 0).",
    )
```

Update `load_grid_insar` signature:

```python
def load_grid_insar(
    nc_path: Path,
    apply_cumsum: bool = True,
    displacement_var: str = "displacement",
    depth_dim: str = "Depth",
    surface_depth: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

Replace line 149 with:

```python
    disp = ds[displacement_var].sel({depth_dim: surface_depth}).values
```

Pass the new args from `main()` (after line 596):

```python
    insar_cum, x_coords, y_coords, time_labels = load_grid_insar(
        args.grid_nc,
        apply_cumsum=apply_cumsum,
        displacement_var=args.displacement_var,
        depth_dim=args.insar_depth_dim,
        surface_depth=args.insar_surface_depth,
    )
```

---

### Issue 9 — Hardcoded NetCDF coordinate names `"X"`, `"Y"`, `"Time"` (Critical)

**Lines 152–161**:
```python
    x_coords = ds["X"].values.copy()
    y_coords = ds["Y"].values.copy()
    if "month_label" in ds:
        time_labels = ds["month_label"].values.astype(str)
    else:
        time_labels = np.array(
            [str(t)[:7] for t in ds["Time"].values], dtype=str
        )
```

**Why it is a problem**: The dimension names `"X"`, `"Y"`, and `"Time"` are hardcoded.
A NetCDF with dimensions named `"lon"`, `"lat"`, `"time"` (lower case) or `"x"`,
`"y"`, `"t"` will raise `KeyError`. The fallback for `"Time"` already exists but the
primary names are still fixed.

**Fix**: Add arguments to `load_grid_insar`:

```python
def load_grid_insar(
    nc_path: Path,
    apply_cumsum: bool = True,
    displacement_var: str = "displacement",
    depth_dim: str = "Depth",
    surface_depth: int = 0,
    x_dim: str = "X",
    y_dim: str = "Y",
    time_dim: str = "Time",
    time_label_var: str = "month_label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

Replace lines 152–161 with:

```python
    x_coords = ds[x_dim].values.copy()
    y_coords = ds[y_dim].values.copy()
    if time_label_var in ds:
        time_labels = ds[time_label_var].values.astype(str)
    else:
        time_labels = np.array(
            [str(t)[:7] for t in ds[time_dim].values], dtype=str
        )
```

Add matching CLI arguments to `parse_args()`:

```python
    p.add_argument("--x-dim",         default="X",            type=str,
        help="Name of the X (easting) dimension in the grid NetCDF.")
    p.add_argument("--y-dim",         default="Y",            type=str,
        help="Name of the Y (northing) dimension in the grid NetCDF.")
    p.add_argument("--time-dim",      default="Time",         type=str,
        help="Name of the time dimension in the grid NetCDF.")
    p.add_argument("--time-label-var",default="month_label",  type=str,
        help="Name of the string time-label variable in the grid NetCDF (optional).")
```

---

### Issue 10 — GP kernel bounds hardcoded (Known Issue #2, Important)

**Lines 225–228**:
```python
        kernel = (
            Matern(nu=2.5, length_scale=5.0, length_scale_bounds=(1.0, 50.0))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 0.1))
        )
```

**Why it is a problem**: The length scale bounds `(1.0, 50.0)` are in kilometres and
are appropriate for the real CRFP domain (~60 km wide). For synthetic data with
stations separated by tens of metres, the optimiser cannot find a length scale below
1 km and gets stuck at the lower bound, producing a constant prediction and
zero predictive variance. This makes the GP CV appear to work (no exception) but
all predictions are wrong.

**Fix**: Add CLI arguments and config entries for kernel hyperparameters.

Add to `parse_args()`:

```python
    p.add_argument("--gp-length-scale",       default=5.0,  type=float,
        help="Initial GP length scale in km (default: 5.0).")
    p.add_argument("--gp-length-scale-min",   default=0.01, type=float,
        help="Lower bound for GP length scale optimisation in km (default: 0.01).")
    p.add_argument("--gp-length-scale-max",   default=200.0, type=float,
        help="Upper bound for GP length scale optimisation in km (default: 200.0).")
    p.add_argument("--gp-noise-level",        default=1e-3,  type=float,
        help="Initial GP noise level (default: 1e-3).")
    p.add_argument("--gp-noise-level-min",    default=1e-8,  type=float,
        help="Lower bound for GP noise level optimisation (default: 1e-8).")
    p.add_argument("--gp-noise-level-max",    default=1.0,   type=float,
        help="Upper bound for GP noise level optimisation (default: 1.0).")
```

Add these same keys to `pipeline_config.ini` under `[GP]`:

```ini
[GP]
n_restarts          = 10
std_threshold       = 0.3
gp_length_scale     = 5.0
gp_length_scale_min = 0.01
gp_length_scale_max = 200.0
gp_noise_level      = 0.001
gp_noise_level_min  = 1.0e-8
gp_noise_level_max  = 1.0
```

Update the `fit_gp_per_layer` function signature:

```python
def fit_gp_per_layer(
    depth_weights: np.ndarray,
    x_train_km: np.ndarray,
    y_train_km: np.ndarray,
    x_pred_km: np.ndarray,
    y_pred_km: np.ndarray,
    n_restarts: int = 10,
    gp_length_scale: float = 5.0,
    gp_length_scale_bounds: tuple[float, float] = (0.01, 200.0),
    gp_noise_level: float = 1e-3,
    gp_noise_level_bounds: tuple[float, float] = (1e-8, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
```

Replace lines 225–228 inside `fit_gp_per_layer`:

```python
        kernel = (
            Matern(
                nu=2.5,
                length_scale=gp_length_scale,
                length_scale_bounds=gp_length_scale_bounds,
            )
            + WhiteKernel(
                noise_level=gp_noise_level,
                noise_level_bounds=gp_noise_level_bounds,
            )
        )
```

In `main()`, pass these to `fit_gp_per_layer`:

```python
    mean_valid, std_valid = fit_gp_per_layer(
        depth_weights=depth_weights,
        x_train_km=x_train_km,
        y_train_km=y_train_km,
        x_pred_km=x_all_km,
        y_pred_km=y_all_km,
        n_restarts=args.n_restarts,
        gp_length_scale=args.gp_length_scale,
        gp_length_scale_bounds=(args.gp_length_scale_min, args.gp_length_scale_max),
        gp_noise_level=args.gp_noise_level,
        gp_noise_level_bounds=(args.gp_noise_level_min, args.gp_noise_level_max),
    )
```

---

### Issue 11 — Temporal RMSE assignment hardcodes fold time boundaries (Critical)

**Lines 292–313**, `assign_temporal_rmse`:
```python
    fold_boundaries = {
        1: (0, 59),    # fold 1 covers time 0–59
        2: (60, 71),   # fold 2 covers time 60–71
        3: (72, None), # fold 3 covers 72 onward
    }
```

**Why it is a problem**: These time boundaries are derived from the hardcoded
`FOLDS` list. The fold 1 boundary `(0, 59)` covers 60 time steps, which matches
the synthetic InSAR grid (132 time steps; different from the 84-epoch MLCW CSV).
But if the grid has a different number of time steps, the last fold's `None` does
extend correctly, while folds 1 and 2 are wrong. More critically, if the number of
MLCW epochs or grid time steps changes, the mapping between temporal CV fold indices
and grid time indices becomes incorrect.

**Fix**: Make `assign_temporal_rmse` accept the fold definitions and derive time
boundaries from them directly.

Replace the function signature:

```python
def assign_temporal_rmse(
    temporal_cv_df: pd.DataFrame,
    valid_depths_m: list[int],
    n_time: int,
    folds: list[dict] | None = None,
) -> np.ndarray:
```

Replace the `fold_boundaries` block inside `assign_temporal_rmse` (lines 292–296)
with:

```python
    # Build fold boundaries from the actual fold definitions.
    # For each fold, the boundary covers from the start of its validation window
    # to the end of the NEXT fold's training period (or the end of the grid).
    # Fold 1's boundary starts at 0 (covers all early time steps).
    if folds is None:
        # Fallback: use the fold boundaries implied by the temporal CV CSV.
        # Each fold's val_start determines where that fold's RMSE begins.
        fold_ids = sorted(temporal_cv_df["fold"].unique())
        fold_boundaries = {}
        for k, fid in enumerate(fold_ids):
            rows = temporal_cv_df[temporal_cv_df["fold"] == fid]
            # Use val_start from the CSV data if available; otherwise use index.
            if "val_start" in rows.columns:
                t_start = int(rows["val_start"].iloc[0])
            else:
                # Cannot reconstruct boundaries without fold metadata; use proportional split.
                t_start = k * (n_time // len(fold_ids))
            t_end = None if k == len(fold_ids) - 1 else None  # will be set below
            fold_boundaries[fid] = (t_start if k > 0 else 0, t_end)
        # Fill t_end for all but last fold
        fold_id_list = list(fold_boundaries.keys())
        for k in range(len(fold_id_list) - 1):
            fid = fold_id_list[k]
            next_fid = fold_id_list[k + 1]
            fold_boundaries[fid] = (fold_boundaries[fid][0], fold_boundaries[next_fid][0] - 1)
    else:
        # Use the passed fold definitions directly.
        fold_boundaries = {}
        for k, fold_def in enumerate(folds):
            fid = fold_def["fold"]
            t_start = fold_def["val_start"] if k > 0 else 0
            t_end = None if k == len(folds) - 1 else folds[k + 1]["val_start"] - 1
            fold_boundaries[fid] = (t_start, t_end)
```

In `main()`, after computing `folds` (see Issue 11 fix in `cv_temporal_forward.py`),
pass `folds` to `load_cv_errors`. Update `load_cv_errors` signature:

```python
def load_cv_errors(
    spatial_csv: Path | None,
    temporal_csv: Path | None,
    valid_depths_m: list[int],
    n_time: int,
    folds: list[dict] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
```

Pass `folds` to `assign_temporal_rmse` inside `load_cv_errors` (line 361):

```python
        temporal_cv_sigma = assign_temporal_rmse(df, valid_depths_m, n_time, folds=folds)
```

---

### Issue 12 — Output NetCDF attributes hardcode grid resolution and CRS (Nice-to-have)

**Lines 563–568**:
```python
    attrs={
        "description": "GP-interpolated depth weights and layer-wise compaction predictions",
        "CRS": "EPSG:3826 (TWD97 TM2)",
        "grid_resolution_m": 500,
        "created_by": "stage2_gp_interpolation.py",
    },
```

**Why it is a problem**: `grid_resolution_m = 500` and `CRS = "EPSG:3826"` are
hardcoded strings. On a different grid or CRS they will be wrong, misleading any
user who reads the NetCDF metadata.

**Fix**: Add CLI arguments:

```python
    p.add_argument("--grid-resolution-m", default=None, type=float,
        help="Pixel resolution in metres for NetCDF metadata. If not set, "
             "estimated from X coordinate spacing.")
    p.add_argument("--crs",              default="EPSG:3826 (TWD97 TM2)", type=str,
        help="CRS string for NetCDF metadata (default: 'EPSG:3826 (TWD97 TM2)').")
```

In `main()`, after loading the grid (line 599), estimate resolution if not specified:

```python
    if args.grid_resolution_m is not None:
        grid_res = args.grid_resolution_m
    elif len(x_coords) > 1:
        grid_res = float(np.median(np.diff(x_coords)))
    else:
        grid_res = float("nan")
```

Pass `grid_res` and `args.crs` to `build_output_dataset` (new parameters), and
use them in the `attrs` dict:

```python
    attrs={
        "description": "GP-interpolated depth weights and layer-wise compaction predictions",
        "CRS": crs,
        "grid_resolution_m": grid_res,
        "created_by": "stage2_gp_interpolation.py",
    },
```

---

### Issue 13 — `"month_label"` variable name check (Known Issue #1, Important)

**Lines 156–161**:
```python
    if "month_label" in ds:
        time_labels = ds["month_label"].values.astype(str)
    else:
        time_labels = np.array(
            [str(t)[:7] for t in ds["Time"].values], dtype=str
        )
```

**Why it is a problem**: This fallback was added as the fix for the originally reported
issue (Known Issue #1). However, the fallback truncates `Time` coordinate values to 7
characters (`str(t)[:7]`) which works for ISO datetime strings (`"2015-01"` from
`"2015-01-01T00:00:00"`) but fails silently for numeric time coordinates (e.g.,
`"0.00000"` from a CF-convention integer time axis with units like
`"months since 2015-01-01"`).

**Fix**: Add an explicit check for numeric time coordinates and format them properly.

Replace lines 156–161 with:

```python
    _time_label_var = time_label_var  # from function parameter (see Issue 9)
    if _time_label_var in ds:
        time_labels = ds[_time_label_var].values.astype(str)
    else:
        raw_times = ds[time_dim].values
        if np.issubdtype(raw_times.dtype, np.datetime64):
            # ISO datetime: take first 7 characters (YYYY-MM)
            time_labels = np.array([str(t)[:7] for t in raw_times], dtype=str)
        elif np.issubdtype(raw_times.dtype, np.integer) or np.issubdtype(
            raw_times.dtype, np.floating
        ):
            # Numeric time axis (e.g., months since epoch): format as T_000, T_001, ...
            time_labels = np.array(
                [f"T_{int(t):04d}" for t in raw_times], dtype=str
            )
        else:
            # String or object: use as-is
            time_labels = raw_times.astype(str)
```

---

## File: `src/loader.py`

### Issue 14 — Hardcoded NetCDF filename `"grid_pnt_datacube_500m.nc"` (Critical)

**Line 250**:
```python
    nc_path = data_dir / "grid_pnt_datacube_500m.nc"
```

**Why it is a problem**: The NetCDF filename is hardcoded inside `build_real_dataset`.
The pipeline config file already has a `grid_nc` path for `stage2_gp_interpolation.py`,
but `build_real_dataset` ignores it. If the user renames the grid file or uses a
different one, this path will raise `FileNotFoundError` even though the correct path
is in the config.

**Fix**: Add an optional `grid_nc_path` parameter to `build_real_dataset`.

Update the signature (line 185):

```python
def build_real_dataset(
    data_dir: str | Path,
    month_start: int = 0,
    month_end: int | None = None,
    lam: float = 1e-2,
    lam_t: float = 1.0,
    sigma_insar: float = 3.0,
    sigma_well: float = 1.0,
    cumulate: bool = True,
    grid_nc_path: str | Path | None = None,
) -> tuple[SyntheticDataset, SystemConfig, dict[str, Any]]:
```

Replace line 250 with:

```python
    if grid_nc_path is not None:
        nc_path = Path(grid_nc_path)
    else:
        # Search for any *.nc file in data_dir as fallback.
        nc_candidates = list(Path(data_dir).glob("*.nc"))
        if len(nc_candidates) == 1:
            nc_path = nc_candidates[0]
        else:
            # Default to the known filename for backward compatibility.
            nc_path = Path(data_dir) / "grid_pnt_datacube_500m.nc"
```

Pass `grid_nc_path` from all callers. In `cv_spatial_gp.py` (line 370), in
`cv_temporal_forward.py` (line 119), and in `main_real.py` (line 111), add
`grid_nc_path` as a parameter where those call `build_real_dataset`. The value should
come from the `--grid-nc` CLI argument (already in `run_pipeline.py`). This requires
threading the `grid_nc` path through each script's `parse_args`. The simplest change:
add `--grid-nc` to `cv_spatial_gp.py` and `cv_temporal_forward.py` and pass it
through.

---

### Issue 15 — Hardcoded coordinate column names `"Ename"`, `"X_TWD97"`, `"Y_TWD97"` (Important)

**Lines 261, 119–123**:
```python
    station_names: list[str] = stations["Ename"].tolist()
    ...
    for _, row in df.iterrows():
        pidx, gr, gc = coord_to_pixel_index(
            float(row["X_TWD97"]), float(row["Y_TWD97"]), x_coords, y_coords
        )
```

**And line 399**:
```python
        "x_twd97": stations["X_TWD97"].tolist(),
        "y_twd97": stations["Y_TWD97"].tolist(),
```

**Why it is a problem**: The coordinate CSV must have exactly the column names
`"Ename"`, `"X_TWD97"`, and `"Y_TWD97"`. A dataset with `"station_id"`, `"easting"`,
`"northing"` will raise `KeyError`.

**Fix**: Add parameters to `build_real_dataset` and `load_station_coords`:

Update `load_station_coords` signature (line 103):

```python
def load_station_coords(
    coords_path: Path,
    x_coords: npt.NDArray[np.float64],
    y_coords: npt.NDArray[np.float64],
    name_col: str = "Ename",
    x_col: str = "X_TWD97",
    y_col: str = "Y_TWD97",
) -> pd.DataFrame:
```

Replace lines 119–123 with:

```python
    for _, row in df.iterrows():
        pidx, gr, gc = coord_to_pixel_index(
            float(row[x_col]), float(row[y_col]), x_coords, y_coords
        )
```

Add corresponding parameters to `build_real_dataset` and thread them through:

```python
def build_real_dataset(
    ...
    station_name_col: str = "Ename",
    station_x_col: str = "X_TWD97",
    station_y_col: str = "Y_TWD97",
) -> ...:
```

Replace lines 261, 399 with references to these parameters:

```python
    station_names: list[str] = stations[station_name_col].tolist()
    ...
        "x_twd97": stations[station_x_col].tolist(),
        "y_twd97": stations[station_y_col].tolist(),
```

---

### Issue 16 — Hardcoded `"Month_"` prefix in CSV column detection (Important)

**Lines 300, 154**:
```python
    all_month_cols = sorted([c for c in sample_df.columns if c.startswith("Month_")])
```

and inside `find_valid_mlcw_depths` (line 151):
```python
        month_cols = [c for c in df.columns if c.startswith("Month_")]
```

**Why it is a problem**: Both functions assume the month columns start with `"Month_"`.
A dataset using `"month_"`, `"M_"`, or any other prefix will return zero month columns,
causing the inversion to have zero epochs.

**Fix**: Add a `month_col_prefix` parameter to `build_real_dataset` and
`find_valid_mlcw_depths`.

Update `find_valid_mlcw_depths` signature:

```python
def find_valid_mlcw_depths(
    csv_dir: Path,
    station_names: list[str],
    month_col_prefix: str = "Month_",
) -> list[int]:
```

Replace line 151 with:

```python
        month_cols = [c for c in df.columns if c.startswith(month_col_prefix)]
```

Update `build_real_dataset` signature:

```python
    month_col_prefix: str = "Month_",
```

Replace line 300 with:

```python
    all_month_cols = sorted(
        [c for c in sample_df.columns if c.startswith(month_col_prefix)]
    )
```

Thread `month_col_prefix` through the call to `find_valid_mlcw_depths` (line 292):

```python
    valid_depths = find_valid_mlcw_depths(csv_dir, station_names, month_col_prefix)
```

---

### Issue 17 — Hardcoded minimum InSAR fraction threshold `0.50` (Important)

**Line 266**:
```python
    _min_insar_fraction = 0.50
```

**Why it is a problem**: Stations with fewer than 50% valid InSAR months are silently
excluded. For short synthetic datasets this threshold might exclude all stations. The
value is not exposed to the user anywhere.

**Fix**: Add a `min_insar_fraction` parameter to `build_real_dataset`:

```python
def build_real_dataset(
    ...
    min_insar_fraction: float = 0.50,
) -> ...:
```

Replace line 266 with:

```python
    _min_insar_fraction = min_insar_fraction
```

Add a corresponding config key in `pipeline_config.ini` under `[Inversion]`:

```ini
# Minimum fraction of valid InSAR months for a station to be included.
# Stations below this threshold are excluded (default: 0.50 = 50%).
min_insar_fraction = 0.50
```

Read it in `run_pipeline.py` and pass it as a CLI argument to `main_real.py` and
`cv_spatial_gp.py`.

---

### Issue 18 — Hardcoded CSV filename pattern `{name}_insar_mlcw.csv` in loader (Important)

**Lines 270, 299, 332, 334**:
```python
        p = csv_dir / f"{name}_insar_mlcw.csv"
        ...
    sample_csv_path = csv_dir / f"{station_names[0]}_insar_mlcw.csv"
        ...
        csv_path = csv_dir / f"{name}_insar_mlcw.csv"
```

**Why it is a problem**: The filename pattern is repeated four times inside
`build_real_dataset`. Any dataset with a different naming convention breaks all four.

**Fix**: Add a `csv_name_pattern` parameter:

```python
def build_real_dataset(
    ...
    csv_name_pattern: str = "{name}_insar_mlcw.csv",
) -> ...:
```

Replace all four occurrences with:

```python
        p = csv_dir / csv_name_pattern.format(name=name)
        ...
    sample_csv_path = csv_dir / csv_name_pattern.format(name=station_names[0])
        ...
        csv_path = csv_dir / csv_name_pattern.format(name=name)
```

---

### Issue 19 — Hardcoded `"Depth"` index column and `0` as InSAR sentinel (Important)

**Lines 334, 337, 340, 345**:
```python
        df = pd.read_csv(csv_path).set_index("Depth")
        if 0 in df.index:
            ...
                d_insar_matrix[s_idx, t_idx] = float(df.loc[0, mc])
        for l_idx, depth in enumerate(valid_depths):
            if depth not in df.index:
```

**Why it is a problem**: The column named `"Depth"` is used as the index; value `0`
identifies the InSAR surface row. A dataset with a different depth column name or
a different sentinel for the surface will silently produce an empty `d_insar_matrix`.

**Fix**: Add parameters `depth_col` and `insar_depth_val`:

```python
def build_real_dataset(
    ...
    depth_col: str = "Depth",
    insar_depth_val: int = 0,
) -> ...:
```

Replace line 334 with:

```python
        df = pd.read_csv(csv_path).set_index(depth_col)
```

Replace line 337 with:

```python
        if insar_depth_val in df.index:
```

Replace line 340 with:

```python
                    d_insar_matrix[s_idx, t_idx] = float(df.loc[insar_depth_val, mc])
```

---

## File: `src/config.py`

### Issue 20 — `SystemConfig.create_real_domain` hardcodes `n_layers=59` and `delta_z=5.0` (Nice-to-have)

**Lines 107–119**:
```python
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
            ...
        )
```

**Why it is a problem**: The docstring and default `n_layers=59` and `delta_z=5.0`
are specific to one particular CRFP parameterisation (5 m layers, 59 layers = 295 m
depth). The actual pipeline uses `n_layers=30` (10 m layers) and `delta_z=1.0`. Any
code that calls `create_real_domain` with no `n_layers` override will produce an
incorrect config. The factory method is misleading.

**Fix**: Remove the factory method entirely, or change the defaults to match the
actual dataset:

```python
    @classmethod
    def create_real_domain(
        cls,
        grid_size: int,
        well_indices: tuple[int, ...],
        n_layers: int = 30,      # 30 layers at 10 m spacing = 0–300 m
        delta_z: float = 1.0,   # 1.0 because G entries already encode mm directly
        **kwargs: float,
    ) -> "SystemConfig":
        """Real-domain config for CRFP: 10 m vertical layers, 0–300 m depth."""
        return cls(
            grid_size=grid_size,
            n_layers=n_layers,
            well_indices=well_indices,
            delta_z=delta_z,
            sigma_well=kwargs.pop("sigma_well", 1.0),
            sigma_insar=kwargs.pop("sigma_insar", 3.0),
            **kwargs,
        )
```

---

## File: `src/system.py`

### Issue 21 — `build_L_spatial` default threshold `15000.0` m (Important)

**In `src/config.py`, line 61**:
```python
    spatial_dist_threshold_m: float = 15000.0
```

**Why it is a problem**: The 15 km threshold is specific to the CRFP domain where
inter-station distances are in the range 5–60 km. For a synthetic test with
stations 500 m apart, all stations will be connected to all others (the threshold
is much larger than the inter-station distances), creating an over-constrained
spatial smoother. For a larger domain, no edges may be found.

**Fix**: The default should be `None` (auto-detect from data range), with a note
in the docstring.

Replace `src/config.py` line 61:

```python
    spatial_dist_threshold_m: float = 0.0
```

Update `src/system.py`, `build_L_spatial`, to accept `threshold = 0.0` as meaning
"auto-detect from median inter-station distance":

```python
    if threshold <= 0.0:
        # Auto-detect: use 3× the median pairwise distance as the threshold.
        # This connects each station to its nearest ~half of all stations.
        pairwise = dist_matrix[dist_matrix > 0]
        if pairwise.size == 0:
            threshold = 1.0  # single station; no edges possible
        else:
            threshold = 3.0 * float(np.median(pairwise))
```

Add a note in `pipeline_config.ini` under `[Inversion]`:

```ini
# Distance threshold (metres) for spatial smoothing between station pairs.
# Pairs within this distance are connected by a smoothness penalty.
# Set to 0 to auto-detect from median inter-station distance × 3.
spatial_dist_threshold_m = 15000.0
```

---

## File: `run_pipeline.py`

### Issue 22 — Windows console encoding failure (Known Issue #4, Critical)

**Line 188 in `run_pipeline.py`**:
```python
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
```

The subprocess itself already uses `encoding="utf-8"` and `errors="replace"`, so the
subprocess runner is protected. However, the Unicode issue occurs when the child
process writes to its own stdout/stderr using the Windows console default encoding
(e.g., `cp950` for Traditional Chinese Windows). The child process is a separate Python
interpreter that has its own encoding for `sys.stdout`.

**Fix**: Set the `PYTHONIOENCODING` environment variable when launching subprocesses.

Replace the `subprocess.run` call:

```python
import os

def run_stage(...) -> bool:
    ...
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    with open(log_path, "w", encoding="utf-8") as log_fh:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        log_fh.write(result.stdout or "")
```

Add `import os` at the top of `run_pipeline.py` (it is not currently imported).

---

### Issue 23 — `--solver` is hardcoded to `"cvxpy"` in Stage 2 command (Important)

**Line 561**:
```python
        "--solver",       "cvxpy",
```

**Why it is a problem**: The solver choice for the main inversion is hardcoded in the
pipeline runner. Users who want to use `"joint"` or `"independent"` must edit
`run_pipeline.py` directly. The config file has no `solver` key.

**Fix**: Add a `solver` key to `pipeline_config.ini`:

```ini
[Inversion]
solver = cvxpy
```

Read it in `run_pipeline.py`:

```python
    solver = cfg.get("Inversion", "solver", fallback="cvxpy")
```

Replace line 561 with:

```python
        "--solver",       solver,
```

---

### Issue 24 — `cv_spatial_gp.py` always uses `month_start=0` and `month_end=None` (Important)

**`cv_spatial_gp.py` line 372**:
```python
    full_dataset, full_config, full_meta = build_real_dataset(
        data_dir=args.data_dir,
        month_start=0,
        month_end=None,
        ...
    )
```

**Why it is a problem**: `month_start` and `month_end` are hardcoded. If the user runs
the main inversion with a restricted time range (e.g., `--month-start 12
--month-end 71`), the spatial CV will use a different time range and produce
depth_weights from a different period than the reference inversion. The LOSO error
metrics will be computed on incompatible references.

**Fix**: Add `--month-start` and `--month-end` to `cv_spatial_gp.py`:

In `parse_args()` (after line 360):

```python
    p.add_argument("--month-start", default=0, type=int,
        help="First month index (0-based) to include (default: 0).")
    p.add_argument("--month-end", default=None, type=int,
        help="Last month index (0-based, inclusive) to include (default: all).")
```

In `main()`, replace line 372:

```python
    full_dataset, full_config, full_meta = build_real_dataset(
        data_dir=args.data_dir,
        month_start=args.month_start,
        month_end=args.month_end,
        ...
    )
```

In `run_pipeline.py`, add `--month-start` and `--month-end` to `cmd_spatial_cv`
(after line 592):

```python
        "--month-start",   str(month_start),
        *(["--month-end", month_end_raw] if month_end_raw else []),
```

---

### Issue 25 — `cv_temporal_forward.py` always uses `month_start=0` (Important)

**`cv_temporal_forward.py` line 120 inside `run_training_fold`**:
```python
    dataset, config, meta = build_real_dataset(
        data_dir=data_dir,
        month_start=0,
        month_end=train_month_end,
        ...
    )
```

**Why it is a problem**: Same as Issue 24. Training always starts from month 0, even
if the user has configured a different start in the config file.

**Fix**: Add `month_start: int = 0` parameter to `run_training_fold`:

```python
def run_training_fold(
    data_dir: Path,
    train_month_end: int,
    lam: float,
    lam_t: float,
    sigma_insar: float = 3.0,
    sigma_well: float = 1.0,
    month_start: int = 0,
) -> tuple[np.ndarray, dict]:
```

Replace line 120:

```python
    dataset, config, meta = build_real_dataset(
        data_dir=data_dir,
        month_start=month_start,
        month_end=train_month_end,
        ...
    )
```

Add `--month-start` argument to `cv_temporal_forward.py`'s `parse_args()` and pass
it in `main()` when calling `run_training_fold`.

In `run_pipeline.py`, add `--month-start` to `cmd_temporal_cv` (after line 619).

---

### Issue 26 — Stage 6 skip sentinel is a specific PNG filename (Nice-to-have)

**Line 712**:
```python
        skip_if=vis_dir / "inversion" / "deep_residual_scatter.png",
```

**Why it is a problem**: Stage 6 is considered complete only if this specific PNG
exists. If `visualise_results.py` is changed or produces a different set of output
files, the skip logic will always re-run Stage 6 even when it is complete, or always
skip it even when the PNG was deleted.

**Fix**: Use the visualisation directory itself as the sentinel, or a dedicated
marker file.

Replace line 712:

```python
        skip_if=vis_dir / ".stage6_complete",
```

At the end of `visualise_results.py`'s `main()`, write the marker:

```python
    (vis_dir / ".stage6_complete").touch()
```

---

## File: `pipeline_config.ini` and `synthetic_config.ini`

### Issue 27 — Config files have no `[Data]` section for format parameters (Important)

**Current state**: Both config files have `[Paths]`, `[Inversion]`, `[Tuning]`,
`[GP]`, `[CV]`, and `[Report]` sections, but no section for configuring CSV/NetCDF
format details (column names, variable names, depth range, etc.).

**Fix**: Add a `[Data]` section to both config files:

```ini
[Data]
# Name of the index column in each station CSV file.
depth_col            = Depth

# Value in depth_col that identifies the InSAR surface row (not a MLCW layer).
insar_depth_val      = 0

# Prefix of the month columns in each station CSV file.
month_col_prefix     = Month_

# Pattern for CSV filename. {name} will be replaced by the station name.
csv_name_pattern     = {name}_insar_mlcw.csv

# Column names in mlcw_station_coordinates.csv.
station_name_col     = Ename
station_x_col        = X_TWD97
station_y_col        = Y_TWD97

# Name of the displacement variable in the grid NetCDF.
displacement_var     = displacement

# Name of the depth dimension in the grid NetCDF (used to select the surface slice).
insar_depth_dim      = Depth

# Value of the depth dimension that identifies the InSAR surface.
insar_surface_depth  = 0

# Names of the spatial and time dimensions in the grid NetCDF.
x_dim                = X
y_dim                = Y
time_dim             = Time

# Optional string-label variable in the grid NetCDF (set to empty to disable).
time_label_var       = month_label
```

Read these in `run_pipeline.py` and pass them as CLI arguments to each stage script.

---

## File: `src/loader.py` — Additional Issue

### Issue 28 — `get_grid_specs` always expects `"X"`, `"Y"`, `"Depth"` dimension names (Critical)

**Lines 72–78**:
```python
    specs: dict[str, Any] = {
        "x_coords": ds["X"].values.copy(),
        "y_coords": ds["Y"].values.copy(),
        "depths": ds["Depth"].values.copy(),
        "grid_rows": int(ds.sizes["Y"]),
        "grid_cols": int(ds.sizes["X"]),
    }
```

**Why it is a problem**: `get_grid_specs` is called from `build_real_dataset` and
hardcodes the dimension names `"X"`, `"Y"`, `"Depth"`. This duplicates the problem
described in Issue 9.

**Fix**: Add parameters to `get_grid_specs`:

```python
def get_grid_specs(
    nc_path: Path,
    x_dim: str = "X",
    y_dim: str = "Y",
    depth_dim: str = "Depth",
) -> dict[str, Any]:
```

Replace lines 72–78 with:

```python
    specs: dict[str, Any] = {
        "x_coords": ds[x_dim].values.copy(),
        "y_coords": ds[y_dim].values.copy(),
        "depths": ds[depth_dim].values.copy() if depth_dim in ds else np.array([]),
        "grid_rows": int(ds.sizes[y_dim]),
        "grid_cols": int(ds.sizes[x_dim]),
    }
```

---

## File: `src/solvers_temporal.py`

### Issue 29 — OSQP solver tolerance and iteration count are hardcoded (Nice-to-have)

**Lines 219–224**:
```python
    problem.solve(
        solver=cp.OSQP,
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=10000,
    )
```

**Why it is a problem**: For large problems (many stations, long time series) OSQP
may not converge within 10,000 iterations, producing `"optimal_inaccurate"` status.
For small problems, tighter tolerances could be used. These are invisible to the user.

**Fix**: Expose these as parameters in `solve_joint_spacetime_cvxpy`:

```python
def solve_joint_spacetime_cvxpy(
    dataset: SyntheticDataset,
    x_twd97: list[float] | np.ndarray,
    y_twd97: list[float] | np.ndarray,
    solver_eps_abs: float = 1e-5,
    solver_eps_rel: float = 1e-5,
    solver_max_iter: int = 10000,
) -> np.ndarray:
```

Replace lines 219–224 with:

```python
    problem.solve(
        solver=cp.OSQP,
        verbose=False,
        eps_abs=solver_eps_abs,
        eps_rel=solver_eps_rel,
        max_iter=solver_max_iter,
    )
```

Add matching config keys to `pipeline_config.ini` under `[Inversion]`:

```ini
solver_eps_abs  = 1.0e-5
solver_eps_rel  = 1.0e-5
solver_max_iter = 10000
```

---

## Summary Table

| # | File | Line(s) | Category | Priority |
|---|---|---|---|---|
| 1 | `cv_temporal_forward.py` | 63–67 | Hardcoded fold boundaries | **Critical** |
| 2 | `cv_temporal_forward.py` | 178–179 | Hardcoded `"Month_"` prefix in validation labels | **Important** |
| 3 | `cv_temporal_forward.py` | 189 | Hardcoded CSV filename pattern | **Important** |
| 4 | `cv_temporal_forward.py` | 192–200 | Hardcoded `"Depth"` index and `0` as InSAR row | **Important** |
| 5 | `tune_hyperparams.py` | 53–55 | Hardcoded fold-3 globals | **Critical** |
| 6 | `tune_hyperparams.py` / configs | 71 | Default `lam_t` floor = 0.1 | **Important** |
| 7 | `tune_hyperparams.py` | 57–68 | Global mutation via `update_fold_split` | Nice-to-have |
| 8 | `stage2_gp_interpolation.py` | 149 | Hardcoded NetCDF variable `"displacement"` | **Critical** |
| 9 | `stage2_gp_interpolation.py` | 152–161 | Hardcoded NetCDF dims `"X"`, `"Y"`, `"Time"` | **Critical** |
| 10 | `stage2_gp_interpolation.py` | 225–228 | GP kernel bounds hardcoded for 60 km domain | **Important** |
| 11 | `stage2_gp_interpolation.py` | 292–313 | Temporal RMSE assignment hardcodes fold time ranges | **Critical** |
| 12 | `stage2_gp_interpolation.py` | 563–568 | Output NetCDF metadata hardcodes resolution and CRS | Nice-to-have |
| 13 | `stage2_gp_interpolation.py` | 156–161 | Numeric time axis not handled in fallback | **Important** |
| 14 | `src/loader.py` | 250 | Hardcoded NetCDF filename `"grid_pnt_datacube_500m.nc"` | **Critical** |
| 15 | `src/loader.py` | 261, 119–123, 399 | Hardcoded coordinate CSV column names | **Important** |
| 16 | `src/loader.py` | 300, 154 | Hardcoded `"Month_"` prefix in column detection | **Important** |
| 17 | `src/loader.py` | 266 | Hardcoded 50% InSAR coverage threshold | **Important** |
| 18 | `src/loader.py` | 270, 299, 332, 334 | Hardcoded CSV filename pattern (4 occurrences) | **Important** |
| 19 | `src/loader.py` | 334, 337, 340, 345 | Hardcoded `"Depth"` column and `0` InSAR sentinel | **Important** |
| 20 | `src/config.py` | 107–119 | `create_real_domain` defaults wrong for actual dataset | Nice-to-have |
| 21 | `src/config.py` / `src/system.py` | 61 | Spatial threshold 15 km hardcoded | **Important** |
| 22 | `run_pipeline.py` | 185–192 | Windows encoding issue in subprocess | **Critical** |
| 23 | `run_pipeline.py` | 561 | Solver hardcoded to `"cvxpy"` | **Important** |
| 24 | `cv_spatial_gp.py` | 372 | `month_start=0`, `month_end=None` hardcoded | **Important** |
| 25 | `cv_temporal_forward.py` | 120 | `month_start=0` hardcoded in `run_training_fold` | **Important** |
| 26 | `run_pipeline.py` | 712 | Stage 6 skip sentinel is a specific PNG filename | Nice-to-have |
| 27 | `pipeline_config.ini` / `synthetic_config.ini` | — | No `[Data]` section for format parameters | **Important** |
| 28 | `src/loader.py` | 72–78 | `get_grid_specs` hardcodes `"X"`, `"Y"`, `"Depth"` | **Critical** |
| 29 | `src/solvers_temporal.py` | 219–224 | OSQP tolerances and iteration count hardcoded | Nice-to-have |

---

## Implementation Order

Apply changes in this order to avoid breaking dependent code during implementation:

1. **Issues 1 and 5** together (fold boundary computation): these are coupled because
   `tune_hyperparams.py` imports from `cv_temporal_forward.py`. Implement
   `compute_folds()` in `cv_temporal_forward.py` first, then update
   `tune_hyperparams.py`.

2. **Issues 8, 9, 13, 28** (NetCDF variable names): these are isolated to
   `stage2_gp_interpolation.py` and `src/loader.py`. No cross-file dependencies
   beyond the CLI arguments.

3. **Issues 14, 15, 16, 17, 18, 19** (loader parameters): all are inside
   `src/loader.py`. Add all parameters to `build_real_dataset` first, then update
   callers (`main_real.py`, `cv_spatial_gp.py`, `cv_temporal_forward.py`,
   `tune_hyperparams.py`).

4. **Issue 27** (`[Data]` config section): add after loader parameters are finalised.
   This requires changes only to the two config files and `run_pipeline.py`.

5. **Issues 10, 11** (GP kernel bounds and temporal RMSE assignment): change
   `stage2_gp_interpolation.py` after fold boundaries are finalised (since
   `assign_temporal_rmse` depends on fold definitions).

6. **Issues 2, 3, 4, 24, 25** (remaining CSV format parameters): thread through
   `load_validation_data` and `run_training_fold`.

7. **Issues 6, 22, 23, 29** (config tuning, encoding, solver, OSQP): isolated
   changes to config files and single functions. Apply last.

8. **Issues 12, 20, 21, 26** (nice-to-have): apply any time after the critical and
   important issues are resolved.
