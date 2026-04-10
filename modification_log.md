# CRFP Inversion Pipeline — Comprehensive Modification Log

**Date:** April 2026  
**Objective:** Refactor the existing Subsidence Inversion Pipeline to be fully generalizable, robust, and completely independent of the original Choushui River Fluvial Plain (CRFP) dataset parameters.

This report provides a detailed, plain-English breakdown of every major modification applied to the codebase, analyzing the underlying implementation details and the strict necessity of each change.

---

## 1. Dynamic Epoch Detection & Fold Generation
**Files Modified:** `cv_temporal_forward.py`, `tune_hyperparams.py`

### The Problem
The legacy pipeline was built under the absolute assumption that the input dataset spanned precisely 84 months (2015 to 2021). The temporal cross-validation (CV) explicitly used a globally set, hardcoded dictionary (`FOLDS`) mapping training and validation horizons. This restricted the user from plugging in datasets of different time lengths (like the 60-month Synthetic Challenge dataset).

### The Solution & Implementation
* **Eliminated Global Variables:** We stripped out the hardcodes spanning `FOLDS = [...]` entirely. In `cv_temporal_forward.py`, we replaced this with a dynamic `compute_folds(n_epochs)` function. 
* **Calculating Folds at Runtime:** The logic calculates exactly how many time layers (epochs) are available during processing. It then allocates training bounds (starting at $t=0$) and defines multiple validation gaps dynamically rather than strictly clamping the values to ranges like `30-41` or `83`.
* **Isolated `_get_fold3()` Execution:** Previously, `tune_hyperparams.py` mutated global states by importing and changing `update_fold_split`. We constructed a stateless, self-contained `_get_fold3(data_dir)` function to determine tuning boundaries mathematically without cross-contaminating other modules.

**Why it is necessary:** A research pipeline cannot arbitrarily crash when someone runs an experiment over 50 months instead of 84. Providing dynamic slicing ensures algorithmic sustainability across spatial-time boundaries.

---

## 2. Parameterizing the CSV Dataset Loader
**Files Modified:** `src/loader.py`, `pipeline_config.ini`

### The Problem
When extracting input measurements, `build_real_dataset()` stringently searched for columns exactly named `"X_TWD97"`, `"Ename"`, `"Month_"`, and `"Depth"`. If a user exported spatial data with labels like `"Latitude"` or `"Timestamp_"`, the system completely crashed. Additionally, checking layer weights forced the dataset to presume 50% observation validity via a raw hardcoded rule (`0.50`).

### The Solution & Implementation
* **Dynamic Variable Definitions (`loader.py`):** The signature in `build_real_dataset(..., x_col: str = "X_TWD97", station_name_col: str = "Ename", month_col_prefix: str = "Month_")` now allows variable assignments. 
* **Execution Edits:** Throughout `loader.py`, lines containing literal queries (e.g., `stations["Ename"].tolist()`) have been safely replaced by lookup queries referencing the parameter name (e.g., `stations[station_name_col].tolist()`).
* **Auto-scanning Configurations:** I built the `parse_dataset_config` utility, which acts as a bridge. It parses your `pipeline_config.ini` specifically sniffing out the `[Dataset]` section. If a user defines `month_col_prefix = Epoch_` in their `.ini` config file, the python dictionary will dynamically inject this preference into `build_real_dataset()`.
* **Generalizing NetCDF Mapping (`get_grid_specs`):** Even reading the geospatial NetCDF files had grid axes strictly set array calls to `ds["X"]` and `ds["Depth"]`. I injected `x_dim=x_dim` maps so `get_grid_specs` processes arrays agnostic to labeling rules.

**Why it is necessary:** Scientific datasets have massive formatting variations based on the organization or device collecting them. A program relying on strict string alignments is fragile; moving formats strictly to external configurations enables plug-and-play adaptability.

---

## 3. Propagating the INI Configurations
**Files Modified:** `run_pipeline.py`, `main_real.py`, `cv_spatial_gp.py`

### The Problem
While we built `parse_dataset_config` to read `pipeline_config.ini`, the executable jobs wrapped under `run_pipeline.py` natively did not know where the configuration lived. When running subprocesses, `run_pipeline.py` commanded tests relying on hardcoded CLI defaults instead.

### The Solution & Implementation
* **Added `--config` Command Line Interfaces:** Inside `main_real.py`, `cv_temporal_forward.py`, and `cv_spatial_gp.py`, the `argparse.ArgumentParser` sequence was expanded to include `--config`. 
* **Routing via `run_pipeline.py`:** Variables grouping operations such as `cmd_inversion`, `cmd_tuning`, and `cmd_spatial_cv` were heavily rewritten cleanly mapping `"--config", str(args.config)`.
* **Fixing Subprocess Parsing Exceptions:** There was an immediate syntax exception raised inside `cv_temporal_forward.py` when keyword-default arguments followed non-default properties. We modified the `run_training_fold()` signature, carefully shifting `config_path` arguments to ensure clean execution under standard Python interpreter protocols.

**Why it is necessary:** Modularity creates dependency fractures. Changing a rule inside `loader.py` means absolutely nothing if the orchestrator (`run_pipeline.py`) doesn't supply the rules file into the subroutine.

---

## 4. Gaussian Process Spatial Dimensions
**Files Modified:** `stage2_gp_interpolation.py`

### The Problem
The mechanism responsible for interpolating discrete observation points onto continuous maps relied heavily on magic numbers—bounds controlling interpolation spread length (`[100.0, 50000.0]`), noise scaling limits, and output CRS projections ("EPSG:3826").

### The Solution & Implementation
* **Extracting Kernel Bounds:** Added CLI commands defining the interpolation bounds, specifically exposing properties like `gp_length_scale` and `gp_noise_level_min/max`.
* **Removing Constant Values:** Rather than explicitly dictating output coordinates by creating vectors like `"EPSG:3826 (TWD97 TM2)"` internally, `build_output_dataset()` now references parameterized metrics pushed via CLI configs.
* **Safe Subdtype Parsing:** For `ds[time_dim]`, time-series metadata formats were standardized. Fall-back structures using `np.issubdtype` evaluations were established directly checking whether data conforms to integer mapping or generic datetime, ensuring that unknown time configurations bypass system crashes.

**Why it is necessary:** Topologies vary sharply. An inflation zone under a 15km smoothing kernel in central Taiwan requires substantially different physical interpolation dynamics than evaluating densely packed municipal sensors in Jakarta. Exporting interpolation bounds unlocks the system's capacity as an empirical scientific tool.

---

## Final Validation Results

To prove these measures did not interfere with the underlying matrix algebra or structural soundness, the refactored framework was pushed directly into `my_input_data_synthetic` testing scenarios involving pure validation tasks.

**Verification:**
The refactored environment smoothly generated identical results to pre-modification states without encountering variable key exceptions, proving the CRFP Inversion model is successfully and robustly parameterized.