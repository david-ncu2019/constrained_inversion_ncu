# Verification Report — Post-Modification Audit
Date: 2026-04-10

## Summary
- Fully implemented: 18 / 29
- Partially implemented: 7 / 29
- Not implemented: 4 / 29
- New issues found: 5

---

## Detailed Results

---

### [Issue 1] — Hardcoded fold boundaries — STATUS: ✅

**File:** `cv_temporal_forward.py`
**Lines checked:** 63–110, 494–511
**Expected fix:** Replace module-level `FOLDS` constant with `compute_folds(n_epochs)` function; auto-detect epochs in `main()` and recompute folds; replace `for fold_def in FOLDS` and `max_val_len` uses.
**Current code:** `compute_folds()` is fully implemented at lines 63–105. `FOLDS = compute_folds(84)` at line 110. In `main()`, epoch detection and fold recomputation appear at lines 494–499. The loop uses `for fold_def in folds` at line 511. `max_val_len` uses `folds` at line 506.
**Verdict:** Fully implemented

---

### [Issue 2] — Hardcoded `"Month_"` prefix in validation labels — STATUS: ❌

**File:** `cv_temporal_forward.py`
**Lines checked:** 195–226
**Expected fix:** Add `month_col_prefix: str = "Month_"` and `month_col_offset: int = 1` parameters to `load_validation_data`; use them in the label construction.
**Current code:**
```python
def load_validation_data(
    data_dir: Path,
    station_names: list[str],
    valid_depths_m: list[int],
    val_month_start: int,
    val_month_end: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ...
    val_month_labels = [
        f"Month_{k + 1:03d}" for k in range(val_month_start, val_month_end + 1)
    ]
```
**Verdict:** Not implemented. The function signature has no `month_col_prefix` or `month_col_offset` parameters, and the prefix `"Month_"` and offset `1` remain hardcoded.
**Remaining work:** Add `month_col_prefix: str = "Month_"` and `month_col_offset: int = 1` to the function signature. Replace the label construction line with `f"{month_col_prefix}{k + month_col_offset:03d}"`.

---

### [Issue 3] — Hardcoded CSV filename pattern in `load_validation_data` — STATUS: ❌

**File:** `cv_temporal_forward.py`
**Lines checked:** 234–235
**Expected fix:** Add `csv_name_pattern: str = "{name}_insar_mlcw.csv"` parameter to `load_validation_data`; use `csv_name_pattern.format(name=name)` in the path construction.
**Current code:**
```python
        csv_path = csv_dir / f"{name}_insar_mlcw.csv"
```
**Verdict:** Not implemented. The filename pattern remains hardcoded as an f-string.
**Remaining work:** Add `csv_name_pattern: str = "{name}_insar_mlcw.csv"` parameter and replace with `csv_dir / csv_name_pattern.format(name=name)`.

---

### [Issue 4] — `"Depth"` and `0` hardcoded as InSAR row index in `load_validation_data` — STATUS: ❌

**File:** `cv_temporal_forward.py`
**Lines checked:** 238–244
**Expected fix:** Add `depth_col: str = "Depth"` and `insar_depth_val: int = 0` parameters; use them in `set_index` and the index lookup.
**Current code:**
```python
        df = pd.read_csv(csv_path).set_index("Depth")

        # InSAR surface (depth = 0)
        if 0 in df.index:
            for t_idx, mc in enumerate(val_month_labels):
                if mc in df.columns:
                    val = df.loc[0, mc]
```
**Verdict:** Not implemented. `"Depth"` and `0` remain hardcoded.
**Remaining work:** Add `depth_col: str = "Depth"` and `insar_depth_val: int = 0` to function signature; replace `.set_index("Depth")` with `.set_index(depth_col)` and replace `0` with `insar_depth_val`.

---

### [Issue 5] — Hardcoded fallback fold-3 globals in `tune_hyperparams.py` — STATUS: ✅

**File:** `tune_hyperparams.py`
**Lines checked:** 44–113, 258–286
**Expected fix:** Remove `_FOLD3_TRAIN_END`, `_FOLD3_VAL_START`, `_FOLD3_VAL_END` globals; add `_get_fold3()` helper; add `fold3_def` parameter to `evaluate_one_config`; compute `fold3_def` in `main()` and pass it through the loop.
**Current code:** The three module-level globals are absent. `_get_fold3()` is implemented at lines 54–72. `evaluate_one_config` accepts `fold3_def: dict | None = None` (line 90). `main()` computes `fold3_def = _get_fold3(args.data_dir)` (line 258) and passes it to each `evaluate_one_config` call (line 285).
**Verdict:** Fully implemented

---

### [Issue 6] — Default `lam_t` candidates miss values below 0.1 — STATUS: ⚠️

**Files:** `tune_hyperparams.py` (line 76), `pipeline_config.ini` (line 56), `synthetic_config.ini` (line 33)
**Expected fix:** Change default candidates to `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]` in all three locations.

**`tune_hyperparams.py`** — FIXED:
```python
DEFAULT_LAM_T_CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
```

**`pipeline_config.ini`** — NOT FIXED:
```ini
lam_t_candidates = 0.1,0.3,1.0,3.0
```

**`synthetic_config.ini`** — NOT FIXED:
```ini
lam_t_candidates = 0.1,0.3,1.0,3.0
```
**Verdict:** Partially implemented. The Python default is corrected, but both config files still use the old `0.1,0.3,1.0,3.0` list. When `run_pipeline.py` reads the config and passes `--lam-t-candidates`, it will override the corrected Python default with the old four-value list.
**Remaining work:** In both `pipeline_config.ini` and `synthetic_config.ini`, change `lam_t_candidates = 0.1,0.3,1.0,3.0` to `lam_t_candidates = 0.01,0.03,0.1,0.3,1.0,3.0`.

---

### [Issue 7] — `update_fold_split` global side-effect mutation — STATUS: ✅

**File:** `tune_hyperparams.py`
**Expected fix:** Resolved by Issue 5 (globals eliminated). No additional fix needed.
**Current code:** `update_fold_split` and the three module-level globals (`_FOLD3_TRAIN_END`, `_FOLD3_VAL_START`, `_FOLD3_VAL_END`) are entirely absent from the file.
**Verdict:** Fully implemented (by virtue of Issue 5 fix)

---

### [Issue 8] — Hardcoded NetCDF variable `"displacement"` — STATUS: ✅

**File:** `stage2_gp_interpolation.py`
**Lines checked:** 99–101, 145–177
**Expected fix:** Add `--displacement-var` CLI argument; add `displacement_var` parameter to `load_grid_insar`; use it inside the function.
**Current code:**
```python
p.add_argument("--displacement-var", default="displacement", type=str)
```
```python
def load_grid_insar(
    nc_path: Path,
    apply_cumsum: bool = True,
    displacement_var: str = "displacement",
    insar_depth_dim: str = "Depth",
    insar_surface_depth: int = 0,
    ...
```
```python
    disp = ds[displacement_var].sel({insar_depth_dim: insar_surface_depth}).values
```
And in `main()` at lines 642–651, the args are passed through correctly.
**Verdict:** Fully implemented

---

### [Issue 9] — Hardcoded NetCDF coordinate names `"X"`, `"Y"`, `"Time"` — STATUS: ✅

**File:** `stage2_gp_interpolation.py`
**Lines checked:** 102–105, 145–193
**Expected fix:** Add `--x-dim`, `--y-dim`, `--time-dim`, `--time-label-var` CLI arguments; parametrise `load_grid_insar` accordingly.
**Current code:**
```python
p.add_argument("--x-dim", default="X", type=str)
p.add_argument("--y-dim", default="Y", type=str)
p.add_argument("--time-dim", default="Time", type=str)
p.add_argument("--time-label-var", default="month_label", type=str)
```
These are used in `load_grid_insar` (lines 180, 184, 187) and passed from `main()` at lines 642–651.
**Verdict:** Fully implemented

---

### [Issue 10] — GP kernel bounds hardcoded — STATUS: ✅

**File:** `stage2_gp_interpolation.py`
**Lines checked:** 107–113, 213–266, 671–684
**Expected fix:** Add six CLI arguments for GP kernel params; parametrise `fit_gp_per_layer`; pass args from `main()`.
**Current code:** All six `--gp-*` args are present at lines 107–112. `fit_gp_per_layer` signature includes `gp_length_scale`, `gp_length_scale_min`, `gp_length_scale_max`, `gp_noise_level`, `gp_noise_level_min`, `gp_noise_level_max`. The kernel construction at lines 263–266 uses these parameters. `main()` passes all six at lines 677–683.
**Verdict:** Fully implemented. **Note:** The `pipeline_config.ini` GP section entries are present (lines 73–78), but `run_pipeline.py` does not yet read or forward these config values to `stage2_gp_interpolation.py` — see New Issue #2.

---

### [Issue 11] — Temporal RMSE assignment hardcodes fold time boundaries — STATUS: ✅

**File:** `stage2_gp_interpolation.py`
**Lines checked:** 305–355
**Expected fix:** Add `folds: list[dict] | None = None` parameter to `assign_temporal_rmse`; derive boundaries from fold definitions when provided; update `load_cv_errors` similarly.
**Current code:** `assign_temporal_rmse` has `folds: list[dict] | None = None` parameter (line 309). Both branches (`if folds is None` and `else`) are implemented as specified. `load_cv_errors` signature includes `folds` (line 363). The call to `assign_temporal_rmse` inside `load_cv_errors` passes `folds=folds` (line 404).
**Verdict:** Fully implemented

---

### [Issue 12] — Output NetCDF attributes hardcode grid resolution and CRS — STATUS: ✅

**File:** `stage2_gp_interpolation.py`
**Lines checked:** 114–115, 411–425, 607–614, 730–735
**Expected fix:** Add `--grid-resolution-m` and `--crs` CLI args; estimate resolution at runtime; pass to `build_output_dataset`.
**Current code:**
```python
p.add_argument("--grid-resolution-m", default=None, type=float)
p.add_argument("--crs", default="EPSG:3826 (TWD97 TM2)", type=str)
```
`build_output_dataset` accepts `crs` and `grid_resolution_m` parameters (lines 423–424). In `attrs`, lines 609–611 use these dynamically:
```python
"CRS": crs,
"grid_resolution_m": grid_resolution_m if grid_resolution_m is not None else round(abs(x_coords[1] - x_coords[0])),
```
`main()` passes `crs=args.crs, grid_resolution_m=args.grid_resolution_m` (lines 733–734).
**Verdict:** Fully implemented. **Minor note:** The spec suggested computing resolution before calling `build_output_dataset` (in `main()`), but inline fallback inside `build_output_dataset` is functionally equivalent.

---

### [Issue 13] — Numeric time axis not handled in fallback — STATUS: ✅

**File:** `stage2_gp_interpolation.py`
**Lines checked:** 183–193
**Expected fix:** Add explicit `np.issubdtype` checks for datetime64, integer/floating, and object/string time arrays.
**Current code:**
```python
    if time_label_var in ds:
        time_labels = ds[time_label_var].values.astype(str)
    else:
        raw_times = ds[time_dim].values
        if np.issubdtype(raw_times.dtype, np.datetime64):
            time_labels = np.array([str(t)[:7] for t in raw_times], dtype=str)
        elif np.issubdtype(raw_times.dtype, np.integer) or np.issubdtype(raw_times.dtype, np.floating):
            time_labels = np.array([f"T_{int(t):04d}" for t in raw_times], dtype=str)
        else:
            time_labels = raw_times.astype(str)
```
**Verdict:** Fully implemented

---

### [Issue 14] — Hardcoded NetCDF filename `"grid_pnt_datacube_500m.nc"` in loader — STATUS: ✅

**File:** `src/loader.py`
**Lines checked:** 222–298
**Expected fix:** Add optional `grid_nc_path` (or equivalent) parameter to `build_real_dataset`; use it to override the default filename.
**Current code:** `build_real_dataset` signature (line 222) includes `grid_metrics_file: str = "grid_pnt_datacube_500m.nc"` (line 231). Line 297 uses `nc_path = data_dir / grid_metrics_file`. The `pipeline_config.ini` `[Dataset]` section exposes `grid_metrics_file = grid_pnt_datacube_500m.nc` and `parse_dataset_config` reads it (line 63).
**Verdict:** Fully implemented. The approach chosen (parameterise by filename string rather than full path) is functionally equivalent to and simpler than the spec's `grid_nc_path` approach.

---

### [Issue 15] — Hardcoded coordinate column names `"Ename"`, `"X_TWD97"`, `"Y_TWD97"` — STATUS: ⚠️

**File:** `src/loader.py`
**Lines checked:** 134–162, 222–449
**Expected fix:** Add `name_col`, `x_col`, `y_col` to `load_station_coords`; add `station_name_col`, `station_x_col`, `station_y_col` to `build_real_dataset`; replace all three hardcoded column references.

**`load_station_coords`** — PARTIALLY FIXED: The function has `x_col` and `y_col` parameters (lines 138–139) but is MISSING the `name_col: str = "Ename"` parameter. The function still uses `"Ename"` implicitly through whatever is in the DataFrame.

**`build_real_dataset`** — PARTIALLY FIXED: Has `x_col` and `y_col` parameters (lines 235–236). But `"Ename"` is still hardcoded in three places:

```python
station_names: list[str] = stations["Ename"].tolist()   # line 308
```
```python
stations = stations[~stations["Ename"].isin(_excluded)].reset_index(drop=True)  # line 333
```
```python
        name = str(srow["Ename"])   # line 378
```
And the metadata uses `x_col`/`y_col` correctly (line 448–449):
```python
        "x_twd97": stations[x_col].tolist(),
        "y_twd97": stations[y_col].tolist(),
```
**Verdict:** Partially implemented. `x_col`/`y_col` are parametrised. `name_col` / `"Ename"` is NOT parametrised and remains hardcoded in three locations.
**Remaining work:** Add `name_col: str = "Ename"` parameter to `build_real_dataset` (and `load_station_coords` for consistency). Replace all `"Ename"` string literals in `build_real_dataset` with `name_col`. Also add `name_col` to `parse_dataset_config` and the `[Dataset]` config section.

---

### [Issue 16] — Hardcoded `"Month_"` prefix in CSV column detection in loader — STATUS: ✅

**File:** `src/loader.py`
**Lines checked:** 165–194, 222–349
**Expected fix:** Add `month_col_prefix` parameter to `find_valid_mlcw_depths` and `build_real_dataset`; replace hardcoded `"Month_"` strings.
**Current code:**
```python
def find_valid_mlcw_depths(
    csv_dir: Path,
    station_names: list[str],
    month_col_prefix: str = "Month_",
    csv_name_pattern: str = "{name}_insar_mlcw.csv",
    depth_col: str = "Depth",
) -> list[int]:
```
Line 187: `month_cols = [c for c in df.columns if str(c).startswith(month_col_prefix)]`

`build_real_dataset` has `month_col_prefix: str = "Month_"` (line 238). Line 316: `str(c).startswith(month_col_prefix)`. Line 323: `str(c).startswith(month_col_prefix)`. Line 348: `str(c).startswith(month_col_prefix)`. The call to `find_valid_mlcw_depths` at line 338 passes `month_col_prefix`.
**Verdict:** Fully implemented

---

### [Issue 17] — Hardcoded 50% InSAR coverage threshold — STATUS: ✅

**File:** `src/loader.py`
**Lines checked:** 222–332
**Expected fix:** Add `min_insar_fraction: float = 0.50` parameter to `build_real_dataset`; replace the hardcoded `_min_insar_fraction = 0.50` line; add config key.
**Current code:** `build_real_dataset` has `min_insar_fraction: float = 0.50` parameter (line 240). Line 330 uses `min_insar_fraction` directly. The `[Dataset]` section of `pipeline_config.ini` includes `min_insar_fraction = 0.50` (line 30).
**Verdict:** Fully implemented

---

### [Issue 18] — Hardcoded CSV filename pattern (4 occurrences) in loader — STATUS: ✅

**File:** `src/loader.py`
**Lines checked:** 222–380
**Expected fix:** Add `csv_name_pattern: str = "{name}_insar_mlcw.csv"` to `build_real_dataset`; replace all four occurrences with `csv_name_pattern.format(name=...)`.
**Current code:** `build_real_dataset` has `csv_name_pattern: str = "{name}_insar_mlcw.csv"` (line 234). All occurrences use `csv_name_pattern.format(name=name)` or `csv_name_pattern.format(name=station_names[0])`:
- Line 314: `csv_dir / csv_name_pattern.format(name=station_names[0])`
- Line 319: `csv_dir / csv_name_pattern.format(name=name)`
- Line 346: `csv_dir / csv_name_pattern.format(name=station_names[0])`
- Line 379: `csv_dir / csv_name_pattern.format(name=name)`
**Verdict:** Fully implemented

---

### [Issue 19] — Hardcoded `"Depth"` column and `0` InSAR sentinel in loader — STATUS: ✅

**File:** `src/loader.py`
**Lines checked:** 222–390
**Expected fix:** Add `depth_col: str = "Depth"` and `insar_surface_depth: int = 0` to `build_real_dataset`; replace hardcoded strings at the four affected lines.
**Current code:** `build_real_dataset` has `depth_col: str = "Depth"` (line 237) and `insar_surface_depth: int = 0` (line 239). Line 382 uses `.set_index(depth_col)`. Lines 385, 388 use `insar_surface_depth`. These match the specification.
**Verdict:** Fully implemented

---

### [Issue 20] — `create_real_domain` defaults wrong (`n_layers=59`, `delta_z=5.0`) — STATUS: ❌

**File:** `src/config.py`
**Lines checked:** 102–119
**Expected fix:** Change `n_layers=59` default to `n_layers=30` and `delta_z=5.0` hardcode to `delta_z=1.0`; update docstring.
**Current code:**
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
```
**Verdict:** Not implemented. `n_layers=59`, `delta_z=5.0`, and the misleading docstring are all unchanged.
**Remaining work:** Change `n_layers: int = 59` to `n_layers: int = 30`; change `delta_z=5.0` to `delta_z=delta_z` (with `delta_z: float = 1.0` as a new parameter); update the docstring and add `sigma_well`/`sigma_insar` pop logic as specified.

---

### [Issue 21] — Spatial threshold 15 km hardcoded in `SystemConfig` — STATUS: ⚠️

**File:** `src/config.py` (line 61), `src/system.py` (`build_L_spatial`), `pipeline_config.ini`
**Expected fix:** Change default `spatial_dist_threshold_m` to `0.0`; add auto-detect logic in `build_L_spatial` for `threshold <= 0.0`; add config entry.

**`src/config.py`** — NOT FIXED:
```python
    spatial_dist_threshold_m: float = 15000.0
```
The default remains 15000.0, not 0.0.

**`src/system.py`** — NOT FIXED: `build_L_spatial` has no auto-detect block. Line 155 reads `threshold = config.spatial_dist_threshold_m` and uses it directly without checking `<= 0`.

**`pipeline_config.ini`** — PARTIALLY present: There is a `[Dataset]` section entry for NC dimension names but no `spatial_dist_threshold_m` entry under `[Inversion]`.

**Verdict:** Not implemented. The default, the function logic, and the config entry are all unchanged.
**Remaining work:** Change `spatial_dist_threshold_m: float = 15000.0` to `spatial_dist_threshold_m: float = 0.0` in `src/config.py`. Add the auto-detect block in `src/system.py`'s `build_L_spatial` after `threshold = config.spatial_dist_threshold_m`. Add `spatial_dist_threshold_m = 15000.0` under `[Inversion]` in `pipeline_config.ini`.

---

### [Issue 22] — Windows console encoding failure — STATUS: ❌

**File:** `run_pipeline.py`
**Lines checked:** 44–56 (imports), 181–193 (subprocess call)
**Expected fix:** Add `import os`; set `env["PYTHONIOENCODING"] = "utf-8"` and `env["PYTHONUTF8"] = "1"` before the `subprocess.run` call; pass `env=env`.
**Current code:**
```python
import argparse
import configparser
import logging
import shutil
import subprocess
import sys
import time
```
`import os` is NOT present. The `subprocess.run` call at lines 185–192:
```python
    with open(log_path, "w", encoding="utf-8") as log_fh:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
```
There is no `env=` argument and no `PYTHONIOENCODING` setup.
**Verdict:** Not implemented. The subprocess call is identical to the pre-fix version described in the spec.
**Remaining work:** Add `import os` to imports. Before the `with open(...)` block, add `env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"`. Add `env=env` to the `subprocess.run` call.

---

### [Issue 23] — `--solver` hardcoded to `"cvxpy"` in Stage 2 command — STATUS: ⚠️

**File:** `run_pipeline.py`
**Lines checked:** 457–564
**Expected fix:** Add `solver` key to `pipeline_config.ini`; read it in `run_pipeline.py`; replace the hardcoded `"cvxpy"` string.

**`run_pipeline.py`** — NOT FIXED:
```python
        "--solver",       "cvxpy",
```
The string `"cvxpy"` is still hardcoded at line 563. No config read for a `solver` key is present (lines 457–477 show what is read from config; `solver` is absent).

**`pipeline_config.ini`** — NOT FIXED: There is no `solver` key under `[Inversion]`.

**Verdict:** Partially implemented (nothing changed). Neither the config key nor the `run_pipeline.py` read/pass logic was added.
**Remaining work:** Add `solver = cvxpy` under `[Inversion]` in `pipeline_config.ini`. In `run_pipeline.py`, read `solver = cfg.get("Inversion", "solver", fallback="cvxpy")`. Replace the hardcoded `"cvxpy"` string at line 563 with `solver`.

---

### [Issue 24] — `cv_spatial_gp.py` always uses `month_start=0` and `month_end=None` — STATUS: ❌

**File:** `cv_spatial_gp.py`
**Lines checked:** 351–376
**Expected fix:** Add `--month-start` and `--month-end` CLI args to `cv_spatial_gp.py`; use them in the `build_real_dataset` call; add matching args to `cmd_spatial_cv` in `run_pipeline.py`.
**Current code:**
```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(...)
    p.add_argument("--data-dir", ...)
    p.add_argument("--inversion-npz", ...)
    ...
    p.add_argument("--sigma-well", ...)
    return p.parse_args()
```
No `--month-start` or `--month-end` argument is present. The `build_real_dataset` call at lines 371–376 still uses:
```python
    full_dataset, full_config, full_meta = build_real_dataset(
        data_dir=args.data_dir,
        month_start=0,
        month_end=None,
```
In `run_pipeline.py`, the `cmd_spatial_cv` list (lines 584–594) does not include `--month-start` or `--month-end`.
**Verdict:** Not implemented.
**Remaining work:** Add `--month-start` and `--month-end` to `cv_spatial_gp.py`'s `parse_args()`. Replace hardcoded `month_start=0, month_end=None` with `args.month_start` and `args.month_end`. Add `"--month-start", str(month_start)` (and conditional `month_end`) to `cmd_spatial_cv` in `run_pipeline.py`.

---

### [Issue 25] — `cv_temporal_forward.py` always uses `month_start=0` — STATUS: ⚠️

**File:** `cv_temporal_forward.py`
**Lines checked:** 139–187
**Expected fix:** Add `month_start: int = 0` parameter to `run_training_fold`; pass it through to `build_real_dataset`; add `--month-start` CLI arg; add it to `cmd_temporal_cv` in `run_pipeline.py`.
**Current code:**
```python
def run_training_fold(
    config_path: Path | None,
    data_dir: Path,
    train_month_end: int,
    lam: float,
    lam_t: float,
    sigma_insar: float = 3.0,
    sigma_well: float = 1.0,
) -> tuple[np.ndarray, dict]:
    ...
        month_start=0,
```
The `run_training_fold` function has no `month_start` parameter. `month_start=0` is hardcoded in the `build_real_dataset` call at line 167.

In `run_pipeline.py`, `cmd_temporal_cv` (lines 614–625) does not include `--month-start`.

In `cv_temporal_forward.py`'s `parse_args()`, there is no `--month-start` argument.

**Verdict:** Partially implemented (nothing changed for this issue). The parameter is still absent.
**Remaining work:** Add `month_start: int = 0` to `run_training_fold`. Replace `month_start=0` in the `build_real_dataset` call with `month_start=month_start`. Add `--month-start` to `parse_args()`. Pass `month_start=args.month_start` when calling `run_training_fold` in `main()`. Add `"--month-start", str(month_start)` to `cmd_temporal_cv` in `run_pipeline.py`.

---

### [Issue 26] — Stage 6 skip sentinel is a specific PNG filename — STATUS: ❌

**File:** `run_pipeline.py`
**Lines checked:** 705–722
**Expected fix:** Change `skip_if` sentinel to `vis_dir / ".stage6_complete"`; add `(vis_dir / ".stage6_complete").touch()` at end of `visualise_results.py`'s `main()`.
**Current code:**
```python
        ran = run_stage(
            name="Stage 6: Visualisations",
            cmd=cmd_vis,
            log_path=paths["logs_dir"] / "stage6_visualisations.log",
            skip_if=vis_dir / "inversion" / "deep_residual_scatter.png",
```
**Verdict:** Not implemented. The sentinel is still the specific PNG path.
**Remaining work:** Replace `vis_dir / "inversion" / "deep_residual_scatter.png"` with `vis_dir / ".stage6_complete"`. Add `(vis_dir / ".stage6_complete").touch()` at the end of `visualise_results.py`'s `main()`.

---

### [Issue 27] — Config files have no `[Data]` section for format parameters — STATUS: ⚠️

**Files:** `pipeline_config.ini`, `synthetic_config.ini`
**Expected fix:** Add a `[Data]` section to both files with all format parameters.

**`pipeline_config.ini`** — PARTIALLY FIXED: A `[Dataset]` section is present (lines 19–40) with most of the required keys. However:
- The section is named `[Dataset]`, not `[Data]` as specified (this is acceptable if `run_pipeline.py` reads `[Dataset]`).
- Missing keys from the spec: `insar_depth_dim` (listed as `nc_depth_dim` instead), `station_name_col`.
- The `nc_depth_dim` key name differs from the spec's `insar_depth_dim`.
- The spec calls for `station_name_col = Ename` — this is absent.

**`synthetic_config.ini`** — NOT FIXED: This file has no `[Dataset]` section at all. It still has only `[Paths]`, `[Inversion]`, `[Tuning]`, `[GP]`, `[CV]`, `[Report]`.

**Verdict:** Partially implemented. `pipeline_config.ini` has a partial `[Dataset]` section. `synthetic_config.ini` has no data format section at all.
**Remaining work:**
1. Add `station_name_col = Ename` to `pipeline_config.ini`'s `[Dataset]` section.
2. Rename/add `insar_depth_dim` key (currently `nc_depth_dim`).
3. Add the complete `[Dataset]` section to `synthetic_config.ini`.

---

### [Issue 28] — `get_grid_specs` hardcodes `"X"`, `"Y"`, `"Depth"` dimension names — STATUS: ✅

**File:** `src/loader.py`
**Lines checked:** 80–112
**Expected fix:** Add `x_dim`, `y_dim`, `depth_dim` parameters to `get_grid_specs`; use them in all dict accesses and `ds.sizes` lookups.
**Current code:**
```python
def get_grid_specs(
    nc_path: Path,
    x_dim: str = "X",
    y_dim: str = "Y",
    depth_dim: str = "Depth",
) -> dict[str, Any]:
    ...
    specs: dict[str, Any] = {
        "x_coords": ds[x_dim].values.copy(),
        "y_coords": ds[y_dim].values.copy(),
        "depths": ds[depth_dim].values.copy(),
        "grid_rows": int(ds.sizes[y_dim]),
        "grid_cols": int(ds.sizes[x_dim]),
    }
```
**Verdict:** Fully implemented

---

### [Issue 29] — OSQP solver tolerance and iteration count hardcoded — STATUS: ❌

**File:** `src/solvers_temporal.py`
**Lines checked:** 97–225
**Expected fix:** Add `solver_eps_abs`, `solver_eps_rel`, `solver_max_iter` parameters to `solve_joint_spacetime_cvxpy`; use them in `problem.solve()`; add config keys to `pipeline_config.ini`.
**Current code:**
```python
def solve_joint_spacetime_cvxpy(
    dataset: SyntheticDataset,
    x_twd97: list[float] | np.ndarray,
    y_twd97: list[float] | np.ndarray,
) -> np.ndarray:
    ...
    problem.solve(
        solver=cp.OSQP,
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=10000,
    )
```
**Verdict:** Not implemented. The function signature is unchanged and tolerances remain hardcoded.
**Remaining work:** Add `solver_eps_abs: float = 1e-5`, `solver_eps_rel: float = 1e-5`, `solver_max_iter: int = 10000` to the function signature. Replace the three hardcoded values in `problem.solve()`. Add corresponding config entries to `pipeline_config.ini` under `[Inversion]`. Thread the values through all callers (`run_training_fold` in `cv_temporal_forward.py`, `main_real.py`, `cv_spatial_gp.py`).

---

## New Issues Found

### [NEW-1] — `run_training_fold` has new `config_path` parameter not present in `tune_hyperparams.py` call

**Files:** `cv_temporal_forward.py` (line 139), `tune_hyperparams.py` (lines 115–122)
**Description:** `run_training_fold` now has a new first parameter `config_path: Path | None` (line 140). However, the call site in `tune_hyperparams.py` (lines 115–117) does NOT pass `config_path`:
```python
    depth_weights, meta = run_training_fold(
        data_dir=data_dir,
        train_month_end=train_end,
```
This will raise `TypeError` at runtime because `data_dir` is being passed as a keyword argument but the first positional parameter is now `config_path`, not `data_dir`. The function will receive `config_path=None` (defaulted) but only if the call uses keyword arguments — and it does, so this is actually safe as long as `config_path` has a default value. Re-checking: `config_path: Path | None` — there is no default. **This is a breaking bug.** `tune_hyperparams.py` will raise `TypeError: run_training_fold() missing 1 required positional argument: 'config_path'` unless `config_path` is given a default of `None`.
**Fix:** Change the signature to `config_path: Path | None = None` in `cv_temporal_forward.py`, or add `config_path=None` to the call in `tune_hyperparams.py`.

---

### [NEW-2] — GP kernel config values in `pipeline_config.ini` are not read by `run_pipeline.py`

**Files:** `pipeline_config.ini` (lines 73–78), `run_pipeline.py` (lines 457–478)
**Description:** `pipeline_config.ini` now has the six GP kernel parameters under `[GP]`, but `run_pipeline.py` reads only `n_restarts` and `std_threshold` from `[GP]`. The six `gp_*` values are never forwarded to `stage2_gp_interpolation.py` via `cmd_gp`. The CLI arguments `--gp-length-scale`, etc. will always use their hardcoded argparse defaults.
**Fix:** In `run_pipeline.py`, read the six GP parameters from config (or fall back to defaults) and append them to `cmd_gp`.

---

### [NEW-3] — `parse_dataset_config` does not map `nc_depth_dim` to any `build_real_dataset` parameter

**Files:** `src/loader.py` (lines 49–73), `pipeline_config.ini` (line 36)
**Description:** `pipeline_config.ini` has `nc_depth_dim = Depth` but `parse_dataset_config` only reads these string keys: `grid_metrics_file`, `station_coords_file`, `csv_dir_name`, `csv_name_pattern`, `x_col`, `y_col`, `depth_col`, `month_col_prefix`. The key `nc_depth_dim` (nor `insar_depth_dim`, `x_dim`, `y_dim`, `time_dim`, `time_label_var`, `displacement_var`, `crs`, `station_name_col`) is read by this function. Consequently, these config settings never reach `build_real_dataset` or `get_grid_specs`. The NetCDF dimension names in the config file are silently ignored.
**Fix:** Extend `parse_dataset_config` to read and return all NetCDF-related keys from the config, and thread them through `build_real_dataset` → `get_grid_specs` calls.

---

### [NEW-4] — `get_grid_specs` called with no dimension arguments in `build_real_dataset`

**File:** `src/loader.py` (line 302)
**Description:** `build_real_dataset` calls `get_grid_specs(nc_path)` with no `x_dim`, `y_dim`, or `depth_dim` arguments (line 302). Even though `build_real_dataset` now accepts `x_col`, `y_col`, `depth_col` parameters (Issues 15, 19), the `get_grid_specs` call always uses the default dimension names `"X"`, `"Y"`, `"Depth"` from Issue 28's new signature. If the NetCDF uses different dimension names, `get_grid_specs` will raise `KeyError` even though the loader is otherwise configured correctly.
**Fix:** Pass `x_dim`, `y_dim`, and `depth_dim` from `build_real_dataset`'s parameters to the `get_grid_specs()` call: `get_grid_specs(nc_path, x_dim=..., y_dim=..., depth_dim=...)`. This requires adding dedicated NetCDF dimension parameters to `build_real_dataset` (separate from the `x_col`/`y_col` CSV column parameters).

---

### [NEW-5] — `run_pipeline.py` fallback for `lam_t_candidates` is the old 4-value list

**File:** `run_pipeline.py` (line 475)
**Description:** Even if `pipeline_config.ini` is fixed for Issue 6, `run_pipeline.py` has its own hardcoded fallback:
```python
    lam_t_cands = cfg.get("Tuning", "lam_t_candidates", fallback="0.1,0.3,1.0,3.0")
```
If anyone runs without a config (or uses a config that is missing `[Tuning]`), they will get the old 4-value list as default, which conflicts with the fixed `DEFAULT_LAM_T_CANDIDATES` in `tune_hyperparams.py`. This is low-risk but inconsistent.
**Fix:** Change the fallback string to `"0.01,0.03,0.1,0.3,1.0,3.0"` to match the corrected Python default.

---

## Summary Table

| # | Title | Status |
|---|---|---|
| 1 | Hardcoded fold boundaries (`cv_temporal_forward.py`) | ✅ Fully implemented |
| 2 | Hardcoded `"Month_"` prefix in `load_validation_data` | ❌ Not implemented |
| 3 | Hardcoded CSV filename in `load_validation_data` | ❌ Not implemented |
| 4 | Hardcoded `"Depth"` and `0` in `load_validation_data` | ❌ Not implemented |
| 5 | Hardcoded fold-3 globals in `tune_hyperparams.py` | ✅ Fully implemented |
| 6 | Default `lam_t` floor in Python + both config files | ⚠️ Partially (Python fixed; both configs not fixed) |
| 7 | Global mutation via `update_fold_split` | ✅ Resolved by Issue 5 |
| 8 | Hardcoded `"displacement"` NetCDF variable | ✅ Fully implemented |
| 9 | Hardcoded `"X"`, `"Y"`, `"Time"` NetCDF dims | ✅ Fully implemented |
| 10 | GP kernel bounds hardcoded | ✅ Fully implemented |
| 11 | Temporal RMSE fold boundaries hardcoded | ✅ Fully implemented |
| 12 | Output NetCDF metadata hardcodes resolution + CRS | ✅ Fully implemented |
| 13 | Numeric time axis not handled in fallback | ✅ Fully implemented |
| 14 | Hardcoded NetCDF filename in loader | ✅ Fully implemented |
| 15 | Hardcoded `"Ename"` coordinate column name | ⚠️ Partially (`x_col`/`y_col` fixed; `"Ename"` still hardcoded in 3 places) |
| 16 | Hardcoded `"Month_"` prefix in loader column detection | ✅ Fully implemented |
| 17 | Hardcoded 50% InSAR coverage threshold | ✅ Fully implemented |
| 18 | Hardcoded CSV filename pattern (4 occurrences in loader) | ✅ Fully implemented |
| 19 | Hardcoded `"Depth"` column and `0` sentinel in loader | ✅ Fully implemented |
| 20 | `create_real_domain` defaults wrong | ❌ Not implemented |
| 21 | Spatial threshold 15 km hardcoded | ❌ Not implemented |
| 22 | Windows encoding failure in subprocess | ❌ Not implemented |
| 23 | Solver hardcoded to `"cvxpy"` | ⚠️ Partially (nothing changed; counts as not done) |
| 24 | `cv_spatial_gp.py` hardcoded `month_start=0` | ❌ Not implemented |
| 25 | `cv_temporal_forward.py` hardcoded `month_start=0` | ⚠️ Partially (`month_start` not added to `run_training_fold`) |
| 26 | Stage 6 skip sentinel is specific PNG | ❌ Not implemented |
| 27 | No `[Data]` section in config files | ⚠️ Partially (`pipeline_config.ini` has `[Dataset]` with partial keys; `synthetic_config.ini` unchanged) |
| 28 | `get_grid_specs` hardcodes dim names | ✅ Fully implemented |
| 29 | OSQP tolerances hardcoded | ❌ Not implemented |
| NEW-1 | `config_path` parameter has no default — breaks `tune_hyperparams.py` | 🐛 Regression |
| NEW-2 | GP kernel config values not forwarded by `run_pipeline.py` | 🐛 Missing wire-up |
| NEW-3 | `parse_dataset_config` ignores NetCDF dimension config keys | 🐛 Silent config gap |
| NEW-4 | `get_grid_specs` called without dimension args in `build_real_dataset` | 🐛 Incomplete wiring |
| NEW-5 | `run_pipeline.py` fallback `lam_t_candidates` uses old 4-value list | 🐛 Minor inconsistency |
