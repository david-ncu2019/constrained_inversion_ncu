# CRFP Land Subsidence Inversion Pipeline

## Overview

This repository contains a specialized research pipeline for quantifying land subsidence in the **Choushui River Fluvial Plain (CRFP)**, Taiwan. The project integrates spatially continuous satellite **InSAR** surface displacement data with depth-resolved subsurface compaction measurements from **Multilayer Compaction Monitoring Wells (MLCW)**.

The core analytical framework utilizes a **Joint Space-Time (4D) Inversion** approach to recover high-resolution pseudo-3D compaction profiles. This allows researchers to understand how much each subsurface layer (typically 10m intervals down to 300m) contributes to the observed surface subsidence.

## Key Features

- **Joint Space-Time Inversion**: Solves the entire multi-year time series simultaneously with temporal regularization to reduce noise and ensure physical consistency.
- **Constrained Optimization**: Implements Tikhonov-regularized least squares to balance data fitting with smooth subsurface profiles.
- **Scalable Solvers**: Memory-efficient implementations capable of handling large-scale grids and long-term time series data.
- **Dynamic Dataset Support**: Robust loader that handles varying epoch lengths, station counts, and naming conventions via configuration files.
- **Gaussian Process (GP) Interpolation**: Advanced spatial interpolation (Stage 2) to expand station-point findings to the full InSAR grid.
- **Hyperparameter Optimization**: Integrated support for **Optuna** to automate the search for optimal regularization weights ($\lambda$) and correlation lengths.
- **Diagnostic Suite**: Comprehensive visualization tools for verifying InSAR-MLCW consistency and inspecting inversion results.

## Repository Structure

```text
.
├── configs/            # Pipeline and solver configuration files (.ini)
├── src/                # Core modular source code
│   ├── loader.py       # Dataset ingestion and preprocessing
│   ├── system.py       # Matrix assembly (G, I_wells, L matrices)
│   ├── solvers.py      # Core inversion algorithms
│   ├── spatial_solvers.py # Kriging and GP implementations
│   └── visualize.py    # Plotting and reporting utilities
├── tests/              # Comprehensive unit testing suite
├── docs/               # Technical handoffs, logs, and verification reports
├── scripts/            # Secondary processing and diagnostic scripts
├── tools/              # Standalone utility scripts for uncertainty and plotting
├── main_real.py        # Entry point for running inversion on real-world data
├── run_pipeline.py     # Orchestrator for multi-stage execution
└── tune_hyperparams_optuna.py # Automated Bayesian parameter tuning
```

## Getting Started

### Prerequisites

The project is managed using `uv` or standard Python environments. Key dependencies include:
- `numpy`, `scipy`, `pandas`
- `xarray`, `netCDF4`
- `scikit-learn` (for GP)
- `optuna` (for hyperparameter tuning)
- `geostatspy` (for Kriging analysis)

### Basic Usage

1. **Configure the Environment**:
   Update `configs/pipeline_config.ini` with your local data paths and desired solver parameters.

2. **Run the Full Pipeline**:
   ```bash
   python run_pipeline.py --config configs/pipeline_config.ini
   ```

3. **Hyperparameter Tuning**:
   To find the optimal regularization weights for your dataset:
   ```bash
   python tune_hyperparams_optuna.py --config configs/pipeline_config.ini --n-trials 100
   ```

4. **Standalone Inversion**:
   ```bash
   python main_real.py --solver joint --lam 0.01 --lam-t 3.0
   ```

## Physical Context

The inversion honors the physical constraint that total surface displacement measured by InSAR is the sum of compaction in the monitored 0–300m layers (MLCW) plus unmonitored deep-seated compaction (>300m). The model vector $m$ represents the layer-wise incremental compaction at each station and epoch.

## Citation

If you use this code for your research, please refer to the associated manuscript:
*Monitoring and Modeling Multilayer Subsurface Compaction in the Choushui River Fluvial Plain via Integrated InSAR and MLCW Data.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
