# Project Handoff: CRFP Land Subsidence Inversion — Status for Gemini

## 1. What This Project Is

This project quantifies land subsidence in Taiwan's Choushui River Fluvial Plain (CRFP) by jointly inverting satellite InSAR surface displacement with depth-resolved compaction measurements from multilayer compaction wells (MLCW). The inversion recovers a pseudo-3D subsurface compaction profile — how much each 10 m depth layer contributes to observed surface subsidence — using Tikhonov-regularized least squares. The ultimate output is a `(83 epochs, 153 rows, 117 cols, 30 layers)` NetCDF compaction cube covering the full InSAR grid at 500 m resolution.

---

## 2. Data Sources and Format

### InSAR
- **File**: `my_input_data/grid_pnt_datacube_500m.nc` — shape `(153 Y, 117 X, 31 Depth)` in TWD97/TM2 projected meters (EPSG:3826). **Warning**: The third dimension is spatial depth (0–300 m), NOT time. Used only as a spatial grid template.
- **Time series**: Stored per-station inside the same CSVs as MLCW data (Depth=0 row), monthly incremental, Jan 2015–Dec 2021 (83 months).
- **Sign convention**: Raw CSV negative = subsidence. Loader negates all values → positive = compaction.
- **Units**: mm, monthly incremental (NOT cumulative).

### MLCW
- **Files**: `my_input_data/CSV_files/{STATION}_insar_mlcw.csv` — 39 CSV files. Columns: `Depth`, `Month_001`–`Month_083`.
- **Depth=0**: InSAR surface displacement co-located at station.
- **Depth=10–290**: Layer-wise compaction for that single 10 m layer (NOT cumulative). PCHIP-interpolated from raw sensor measurements.
- **Depth=300**: All NaN; auto-excluded by loader.
- **Sign convention**: Same as InSAR. Loader negates.

### Key Physical Rule
`sum(MLCW 0–300 m layers) ≤ InSAR surface displacement` — deep alluvial sediments below 300 m also compact but are not measured by MLCW.

---

## 3. Codebase Structure

| File | Role |
|---|---|
| `main_real.py` | CLI entry point: load → invert → save `output/real_m_est.npz` |
| `automated_runner.py` | Parameter grid search (2 solvers × 5 lam_t = 10 runs) |
| `src/loader.py` | `build_real_dataset()` → returns `(SyntheticDataset, SystemConfig, metadata)` |
| `src/system.py` | `build_G_matrix()`, `build_I_wells()`, `build_L_matrix()` — forward operators |
| `src/config.py` | `SystemConfig` dataclass, `SyntheticDataset` dataclass |
| `src/solvers_temporal.py` | `solve_joint_spacetime()`, `solve_independent_epochs()` |
| `inspect_mlcw_vs_insar.py` | Diagnostic: 35 PNG figures in `inspection_figs/` |

**NPZ output keys**: `m_est (83,35,30)`, `depth_weights (35,30)`, `cumulative_compaction (35,30)`, `insar_coverage (35,)`, `station_names`, `valid_depths_m`, `full_grid_pixel_indices`, `x_twd97`, `y_twd97`, `grid_rows=153`, `grid_cols=117`.

**Optimal inversion parameters** (from prior grid search, Run 12): `--solver joint --lam-t 3.0 --lam 0.01` (MSE = 14.03 mm²). The current NPZ was produced with default `lam_t=1.0` — **re-run with `--lam-t 3.0` before Stage 2**.

---

## 4. What Was Done This Session

1. New CSV data provided: PCHIP ring-by-ring MLCW compaction for all 39 stations.
2. Loader fix: exclusion filter added for stations with <50% InSAR temporal coverage — dropped **ANNAN** (43%), **JINHU_XIN** (13%), **JIUZHUANG** (19%), **NANGUANG** (47%). 35 stations remain.
3. Inversion ran successfully: 35 stations, 83 epochs, 30 layers, ~22 seconds.
4. Visualization created: 35 PNG figures in `inspection_figs/` comparing cumulative InSAR vs per-depth cumulative MLCW.
5. Data quality finding: `insar_coverage > 1.0` for all 35 stations (mean 587%) — see Section 5.

---

## 5. Current Known Issue: `insar_coverage > 1.0` for All Stations

**Root cause**: The formula in `main_real.py` divides cumulative MLCW layer compaction (summed over 29 layers × 83 epochs) by net InSAR surface displacement (a single scalar per station). These are not the same physical quantity — the sum across layers and months is much larger than net surface displacement, so the ratio is always > 1.

**Worst cases**: ZHENNAN (136×, near-zero net InSAR = unstable denominator), FENGAN (5.6×), XIGANG (5.2×), JIAXING (5.0×).

**The formula needs correction** before `depth_weights` can be used in Stage 2.

---

## 6. The Unresolved Question

**What should `depth_weights[s, l]` represent?** The current formula produces values > 1.0 and is uninterpretable. The most defensible definition:

> `depth_weights[s, l] = cumulative_compaction[s, l] / cumulative_compaction[s].sum()`

i.e., the fraction of total MLCW-measured compaction at station `s` attributable to depth layer `l`. This sums to 1.0 per station and is stable. The fraction of InSAR explained by 0–300 m can be tracked separately as a diagnostic.

This must be resolved before Stage 2.

---

## 7. Next Steps (Stage 2)

**Goal**: Interpolate per-station `depth_weights (35, 29)` to every pixel in the 153×117 InSAR grid using RBF in TWD97 coordinates, then multiply by per-pixel InSAR surface displacement to produce the `(83, 153, 117, 29)` compaction cube.

**Files to create**: `stage2_interpolate.py`, `src/interpolator.py`, `tests/test_interpolator.py`.

---

## 8. Key Gotchas for Gemini

1. **CRS is TWD97/TM2 (EPSG:3826), NOT WGS84**. All spatial operations in projected meters.
2. **Loader negates all values**. Positive in code = compaction/subsidence. Raw CSV negative = subsidence.
3. **MLCW layers are incremental per layer, not cumulative depth profiles**.
4. **NetCDF third dimension = spatial depth, not time**. It is a grid template only.
5. **NetCDF masked arrays**: use `.filled(np.nan)` before numpy calls.
6. **39 CSV files, 35 active stations**. The 4 excluded are still in the coordinates CSV.
7. **`delta_z = 1.0`** in all runs; `G @ m = sum(mm_layer)` directly.
8. **Current NPZ used `lam_t=1.0`**, not the optimal `3.0`. Re-run before Stage 2.
9. **Month_001 = January 2015** (month index 0). Columns are 1-indexed in filenames.
10. **`m_est` in NPZ shape is `(n_epochs, n_stations, n_layers)`** — epoch-major 3D.
