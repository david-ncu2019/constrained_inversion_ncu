"""
run_pipeline.py — Unified automation script for the CRFP subsidence inversion pipeline.

Orchestrates all stages in the correct order, auto-selects hyperparameters from CV,
enforces consistent parameters across stages, and supports resuming from intermediate
outputs.

Usage
-----
  uv run python run_pipeline.py [options]

Options
-------
  --config PATH       Path to pipeline_config.ini (default: pipeline_config.ini)
  --force             Rerun all stages even if outputs exist
  --skip-tuning       Skip Stage 1; require [Tuning] fixed_lam and fixed_lam_t in config
  --skip-cv           Skip Stages 3a and 3b; omit combined uncertainty from output
  --output-dir PATH   Override [Paths] output_dir from config
  --dry-run           Print commands that would be run without executing them

Pipeline Stages
---------------
  1  Tuning       — grid search over lam × lam_t (tune_hyperparams.py)
  2  Inversion    — joint space-time inversion with best params (main_real.py)
  3a Spatial CV   — leave-one-station-out GP CV (cv_spatial_gp.py)
  3b Temporal CV  — forward-chaining CV (cv_temporal_forward.py)
  4  GP Expansion — interpolate depth weights to 500 m grid (stage2_gp_interpolation.py)
  5  Summary      — write pipeline_summary.txt

Outputs (under [Paths] output_dir)
-----------------------------------
  tune_hyperparams/hyperparam_results_full.csv
  tune_hyperparams/hyperparam_summary_table.csv
  best_params.txt
  real_m_est.npz
  cv_spatial_gp/loso_per_layer.csv
  cv_temporal/temporal_cv_per_layer.csv
  depth_weights_gp_grid.nc
  pipeline_summary.txt
  logs/stage_*.log
"""


import argparse
import configparser
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> configparser.ConfigParser:
    """
    Load and validate the pipeline config file.

    Parameters
    ----------
    config_path : Path

    Returns
    -------
    configparser.ConfigParser
    """
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Copy pipeline_config.ini and edit it, or specify a path with --config."
        )
    cfg.read(config_path)
    return cfg


def get_next_output_dir(parent_dir: Path, prefix: str = "output_") -> Path:
    """Find the next available numbered directory: output_001, output_002, etc."""
    if not parent_dir.exists():
        return parent_dir / f"{prefix}001"
    
    existing = [
        d.name for d in parent_dir.glob(f"{prefix}*") if d.is_dir()
    ]
    indices = []
    for name in existing:
        # Expecting prefix followed by digits
        suffix = name[len(prefix):]
        if suffix.isdigit():
            indices.append(int(suffix))
    
    next_idx = max(indices, default=0) + 1
    return parent_dir / f"{prefix}{next_idx:03d}"


def get_paths(cfg: configparser.ConfigParser, output_dir_override: Optional[str]) -> Dict[str, Path]:
    """Resolve all key paths from the config."""
    data_dir = Path(cfg.get("Paths", "data_dir", fallback="my_input_data"))
    
    # If no override is provided, we use the auto-numbering scheme inside data_dir
    if output_dir_override:
        output_dir = Path(output_dir_override)
    else:
        # Default behavior: create output_001, output_002, etc. inside data_dir
        output_dir = get_next_output_dir(data_dir)
        
    grid_nc = Path(cfg.get("Paths", "grid_nc",
                            fallback="my_input_data/grid_pnt_CRFP_500m_vert_IDW_v1.nc"))
    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "grid_nc": grid_nc,
        "logs_dir": output_dir / "logs",
        "tuning_dir": output_dir / "tune_hyperparams",
        "spatial_cv_dir": output_dir / "cv_spatial_gp",
        "temporal_cv_dir": output_dir / "cv_temporal",
        "best_params": output_dir / "best_params.txt",
        "inversion_npz": output_dir / "real_m_est.npz",
        "spatial_csv": output_dir / "cv_spatial_gp" / "loso_per_layer.csv",
        "temporal_csv": output_dir / "cv_temporal" / "temporal_cv_per_layer.csv",
        "tuning_csv": output_dir / "tune_hyperparams" / "hyperparam_results_full.csv",
        "gp_nc": output_dir / "depth_weights_gp_grid.nc",
        "summary": output_dir / "pipeline_summary.txt",
    }


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_stage(
    name: str,
    cmd: List[str],
    log_path: Path,
    skip_if: Optional[Path],
    force: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> bool:
    """
    Run one pipeline stage.

    Parameters
    ----------
    name : str          — human-readable stage name
    cmd : list[str]     — command tokens (passed to subprocess.run)
    log_path : Path     — file to capture stdout + stderr
    skip_if : Path | None — skip if this file exists (and not force)
    force : bool        — if True, ignore skip_if
    dry_run : bool      — print command without executing
    logger : logging.Logger

    Returns
    -------
    bool — True if stage ran, False if skipped
    """
    if not force and skip_if is not None and skip_if.exists():
        logger.info(f"[SKIP] {name} — output exists: {skip_if}")
        return False

    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"[RUN ] {name}")
    logger.info(f"       cmd: {cmd_str}")

    if dry_run:
        print(f"\n  [DRY-RUN] {name}")
        print(f"  cmd: {cmd_str}")
        print(f"  log: {log_path}")
        return True

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as log_fh:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        log_fh.write(result.stdout or "")

    elapsed = time.perf_counter() - t0

    # Echo last 20 lines to console
    lines = (result.stdout or "").splitlines()
    for line in lines[-20:]:
        logger.info(f"       {line}")

    if result.returncode != 0:
        logger.error(f"[FAIL] {name} exited with code {result.returncode}")
        logger.error(f"       Full log: {log_path}")
        raise RuntimeError(
            f"Stage '{name}' failed (exit code {result.returncode}). "
            f"See {log_path} for details."
        )

    logger.info(f"[DONE] {name}  ({elapsed:.0f}s)")
    return True


# ---------------------------------------------------------------------------
# Hyperparameter helpers
# ---------------------------------------------------------------------------


def select_best_params(tuning_csv: Path) -> Tuple[float, float]:
    """
    Read hyperparam_results_full.csv and return (lam, lam_t) with lowest mean_rmse_mm.

    Parameters
    ----------
    tuning_csv : Path

    Returns
    -------
    (lam, lam_t) : tuple[float, float]
    """
    df = pd.read_csv(tuning_csv)
    best_row = df.loc[df["mean_rmse_mm"].idxmin()]
    return float(best_row["lam"]), float(best_row["lam_t"])


def write_best_params(path: Path, lam: float, lam_t: float) -> None:
    path.write_text(f"lam={lam}\nlam_t={lam_t}\n", encoding="utf-8")


def read_best_params(path: Path) -> Tuple[float, float]:
    lines = {
        k.strip(): float(v.strip())
        for k, v in (line.split("=") for line in path.read_text().splitlines() if "=" in line)
    }
    return lines["lam"], lines["lam_t"]


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def generate_summary(
    config_path: Path,
    paths: Dict[str, Path],
    best_lam: float,
    best_lam_t: float,
    stages_run: List[str],
    elapsed_s: float,
    skip_cv: bool,
    logger: logging.Logger,
) -> None:
    """Write output/pipeline_summary.txt with key metrics from each stage."""
    lines = [
        "=== Pipeline Run Summary ===",
        f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Config     : {config_path}",
        f"Output dir : {paths['output_dir']}",
        f"Stages run : {', '.join(stages_run) if stages_run else 'none (all skipped)'}",
        "",
    ]

    lines += [
        "Hyperparameters:",
        f"  lam   = {best_lam}",
        f"  lam_t = {best_lam_t}",
        "",
    ]

    # Tuning metrics
    tuning_csv: Path = paths["tuning_csv"]
    if tuning_csv.exists():
        df_t = pd.read_csv(tuning_csv)
        best_row = df_t.loc[df_t["mean_rmse_mm"].idxmin()]
        lines += [
            "Hyperparameter Tuning (fold 3):",
            f"  Best lam={best_row['lam']:.4g}  lam_t={best_row['lam_t']:.4g}",
            f"  Fold-3 mean RMSE: {best_row['mean_rmse_mm']:.3f} mm",
            f"  Fold-3 max  RMSE: {best_row['max_rmse_mm']:.3f} mm",
            "",
        ]

    # Spatial CV metrics
    spatial_csv: Path = paths["spatial_csv"]
    if spatial_csv.exists():
        df_s = pd.read_csv(spatial_csv)
        mean_rmse = df_s["rmse"].mean()
        worst = df_s.loc[df_s["rmse"].idxmax()]
        lines += [
            "Spatial CV (LOSO):",
            f"  Mean per-layer RMSE : {mean_rmse:.5f} (dimensionless weight units)",
            f"  Worst layer         : {int(worst['depth_m'])} m  RMSE = {worst['rmse']:.5f}",
            "",
        ]
    elif not skip_cv:
        lines += ["Spatial CV: not completed\n"]

    # Temporal CV metrics
    temporal_csv: Path = paths["temporal_csv"]
    if temporal_csv.exists():
        df_tc = pd.read_csv(temporal_csv)
        lines += ["Temporal CV (forward chaining):"]
        for fold_id in sorted(df_tc["fold"].unique()):
            fold_rows = df_tc[df_tc["fold"] == fold_id]
            mean_r = fold_rows["rmse"].mean()
            lines.append(f"  Fold {fold_id} mean RMSE : {mean_r:.3f} mm")
        lines.append("")
    elif not skip_cv:
        lines += ["Temporal CV: not completed\n"]

    # GP expansion
    gp_nc: Path = paths["gp_nc"]
    if gp_nc.exists():
        has_combined = False
        try:
            import xarray as xr
            ds = xr.open_dataset(gp_nc)
            has_combined = "compaction_total_std" in ds
            lc = int(ds["low_confidence"].values.sum()) if "low_confidence" in ds else None
            total_px = ds["low_confidence"].size if "low_confidence" in ds else None
            ds.close()
        except Exception:
            lc = None
            total_px = None

        lines += ["GP Grid Expansion:"]
        lines += [f"  Output : {gp_nc}"]
        if lc is not None and total_px is not None:
            lines += [f"  Low-confidence pixels: {lc} / {total_px} "
                      f"({100*lc/total_px:.1f}%)"]
        lines += [f"  compaction_total_std : {'included' if has_combined else 'NOT included (CV skipped or failed)'}"]
        lines += [""]

    # Anisotropy metrics
    aniso_json = paths["gp_nc"].with_suffix(".anisotropy.json")
    if aniso_json.exists():
        import json
        with open(aniso_json, "r") as f:
            params = json.load(f)
        
        lines += ["Learned Anisotropy (Rotated GPR):"]
        for p in params:
            if "rotation_angle_deg" in p:
                lines += [
                    f"  Layer {p['layer']}: Angle={p['rotation_angle_deg']:.1f}°, "
                    f"Ratio={p['anisotropy_ratio']:.2f}:1, LML={p['log_marginal_likelihood']:.2f}"
                ]
        lines += [""]

    # Honest uncertainty statement
    if not skip_cv and temporal_csv.exists() and spatial_csv.exists():
        df_tc2 = pd.read_csv(temporal_csv)
        fold3 = df_tc2[df_tc2["fold"] == 3]["rmse"].mean()
        df_s2 = pd.read_csv(spatial_csv)
        sp_rmse = df_s2["rmse"].mean()
        lines += [
            "Uncertainty interpretation:",
            (
                f"  Given spatial mismatches (area-averaged InSAR vs point-wise MLCW) "
                f"and temporal extrapolation, the predicted layer compaction carries:"
            ),
            f"  - GP interpolation uncertainty: see depth_weight_std in NetCDF",
            f"  - Spatial CV RMSE (mean across layers): {sp_rmse:.5f} (weight units)",
            f"  - Temporal CV RMSE, fold 3 (best proxy for 2022 gap): {fold3:.3f} mm",
            (
                "  These represent the best achievable performance under current "
                "data limitations."
            ),
            "",
        ]

    lines += [
        f"Total elapsed: {elapsed_s/3600:.2f} h  ({elapsed_s:.0f} s)",
    ]

    summary_text = "\n".join(lines)
    paths["summary"].write_text(summary_text, encoding="utf-8")
    logger.info(f"[DONE] Summary written to {paths['summary']}")
    print("\n" + summary_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified CRFP inversion pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default="pipeline_config.ini", type=Path,
        help="Path to pipeline_config.ini (default: pipeline_config.ini).",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Rerun all stages even if outputs already exist.",
    )
    p.add_argument(
        "--skip-tuning", action="store_true", default=False,
        help="Skip Stage 1 (hyperparameter tuning). "
             "Requires [Tuning] fixed_lam and fixed_lam_t in config.",
    )
    p.add_argument(
        "--skip-cv", action="store_true", default=False,
        help="Skip Stages 3a and 3b (spatial and temporal CV). "
             "The output NetCDF will not include compaction_total_std.",
    )
    p.add_argument(
        "--output-dir", default=None, type=str,
        help="Override [Paths] output_dir from the config file.",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print all commands that would be run without executing them.",
    )
    p.add_argument(
        "--make-plots", action="store_true", default=False,
        help="Run Stage 6 (visualise_results.py) after the summary report.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    t_global_start = time.perf_counter()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    paths = get_paths(cfg, args.output_dir)

    # Create output directories
    for key in ("output_dir", "logs_dir", "tuning_dir", "spatial_cv_dir", "temporal_cv_dir"):
        if not args.dry_run:
            paths[key].mkdir(parents=True, exist_ok=True)

    # ── Copy config to output dir ─────────────────────────────────────────
    if not args.dry_run:
        shutil.copy2(args.config, paths["output_dir"] / "pipeline_config.ini")

    # ── Logging ───────────────────────────────────────────────────────────
    log_fmt = "%(asctime)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *(
                [logging.FileHandler(paths["output_dir"] / "pipeline.log", encoding="utf-8")]
                if not args.dry_run
                else []
            ),
        ],
    )
    logger = logging.getLogger("pipeline")

    # ── Inversion hyperparameters ─────────────────────────────────────────
    sigma_insar = cfg.getfloat("Inversion", "sigma_insar", fallback=3.0)
    sigma_well  = cfg.getfloat("Inversion", "sigma_well",  fallback=1.0)
    month_start = cfg.getint("Inversion",   "month_start", fallback=0)
    month_end_raw = cfg.get("Inversion",    "month_end",   fallback="").strip()

    # GP config
    n_restarts    = cfg.getint("GP", "n_restarts",   fallback=10)
    std_threshold = cfg.getfloat("GP", "std_threshold", fallback=0.3)
    n_gp_restarts = cfg.getint("CV", "n_gp_restarts", fallback=10)
    run_spatial_cv  = cfg.getboolean("CV", "run_spatial_cv",  fallback=True)
    run_temporal_cv = cfg.getboolean("CV", "run_temporal_cv", fallback=True)
    no_plots = cfg.getboolean("Report", "no_plots", fallback=False)
    make_plots_cfg = cfg.getboolean("Report", "make_plots", fallback=False)
    make_plots = args.make_plots or make_plots_cfg

    # Anisotropy config
    gp_mode        = cfg.get("GP", "gp_mode",        fallback="isotropic")
    angle_search   = cfg.get("GP", "angle_search",   fallback="bounded")
    max_anisotropy = cfg.get("GP", "max_anisotropy", fallback="").strip()
    angle_min      = cfg.getfloat("GP", "angle_min", fallback=0.0)
    angle_max      = cfg.getfloat("GP", "angle_max", fallback=180.0)

    # Tuning config
    lam_cands   = cfg.get("Tuning", "lam_candidates",   fallback="0.001,0.01,0.1")
    lam_t_cands = cfg.get("Tuning", "lam_t_candidates", fallback="0.1,0.3,1.0,3.0")
    fixed_lam_str   = cfg.get("Tuning", "fixed_lam",   fallback="").strip()
    fixed_lam_t_str = cfg.get("Tuning", "fixed_lam_t", fallback="").strip()

    # Determine if tuning should be skipped because fixed values are provided
    use_fixed = bool(fixed_lam_str and fixed_lam_t_str)
    if args.skip_tuning and not use_fixed:
        logger.error(
            "--skip-tuning requires [Tuning] fixed_lam and fixed_lam_t to be set in the config."
        )
        sys.exit(1)

    skip_cv = args.skip_cv or (not run_spatial_cv and not run_temporal_cv)

    # ── Python interpreter ────────────────────────────────────────────────
    python = sys.executable

    stages_run: list[str] = []

    if args.dry_run:
        print("\n=== DRY RUN — commands that would be executed ===")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 1: Hyperparameter Tuning
    # ════════════════════════════════════════════════════════════════════════
    if use_fixed or args.skip_tuning:
        best_lam   = float(fixed_lam_str)
        best_lam_t = float(fixed_lam_t_str)
        logger.info(
            f"[SKIP] Stage 1 (Tuning) — using fixed lam={best_lam}, lam_t={best_lam_t}"
        )
    else:
        tuning_skip_cond = paths["tuning_csv"] if not args.force else None
        # Also need best_params.txt to exist for a valid skip
        if (
            not args.force
            and paths["tuning_csv"].exists()
            and paths["best_params"].exists()
        ):
            tuning_skip_cond = paths["tuning_csv"]  # already done
        else:
            tuning_skip_cond = None

        cmd_tuning = [
            python, "tune_hyperparams.py",
            "--config",          str(args.config),
            "--data-dir",        str(paths["data_dir"]),
            "--output-dir",      str(paths["tuning_dir"]),
            "--sigma-insar",     str(sigma_insar),
            "--sigma-well",      str(sigma_well),
            "--lam-candidates",  lam_cands,
            "--lam-t-candidates", lam_t_cands,
            "--no-plot",
        ]
        ran = run_stage(
            name="Stage 1: Hyperparameter Tuning",
            cmd=cmd_tuning,
            log_path=paths["logs_dir"] / "stage1_tuning.log",
            skip_if=tuning_skip_cond,
            force=args.force,
            dry_run=args.dry_run,
            logger=logger,
        )
        if ran:
            stages_run.append("tuning")

        if not args.dry_run:
            best_lam, best_lam_t = select_best_params(paths["tuning_csv"])
            write_best_params(paths["best_params"], best_lam, best_lam_t)
            logger.info(
                f"       Auto-selected: lam={best_lam}, lam_t={best_lam_t}"
            )
        else:
            best_lam, best_lam_t = 0.01, 0.3   # placeholder for dry-run display

    # ════════════════════════════════════════════════════════════════════════
    # Stage 2: Inversion
    # ════════════════════════════════════════════════════════════════════════
    cmd_inversion = [
        python, "main_real.py",
        "--config",       str(args.config),
        "--data-dir",     str(paths["data_dir"]),
        "--output-dir",   str(paths["output_dir"]),
        "--lam",          str(best_lam),
        "--lam-t",        str(best_lam_t),
        "--sigma-insar",  str(sigma_insar),
        "--sigma-well",   str(sigma_well),
        "--month-start",  str(month_start),
        "--solver",       "cvxpy",
    ]
    if month_end_raw:
        cmd_inversion += ["--month-end", month_end_raw]

    ran = run_stage(
        name="Stage 2: Joint Inversion (CVXPY)",
        cmd=cmd_inversion,
        log_path=paths["logs_dir"] / "stage2_inversion.log",
        skip_if=paths["inversion_npz"],
        force=args.force,
        dry_run=args.dry_run,
        logger=logger,
    )
    if ran:
        stages_run.append("inversion")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 3a: Spatial CV (LOSO)
    # ════════════════════════════════════════════════════════════════════════
    if not skip_cv and run_spatial_cv:
        cmd_spatial_cv = [
            python, "cv_spatial_gp.py",
            "--config",         str(args.config),
            "--data-dir",       str(paths["data_dir"]),
            "--inversion-npz",  str(paths["inversion_npz"]),
            "--output-dir",     str(paths["spatial_cv_dir"]),
            "--lam",            str(best_lam),
            "--lam-t",          str(best_lam_t),
            "--sigma-insar",    str(sigma_insar),
            "--sigma-well",     str(sigma_well),
            "--n-restarts",     str(n_gp_restarts),
        ]
        ran = run_stage(
            name="Stage 3a: Spatial CV (LOSO)",
            cmd=cmd_spatial_cv,
            log_path=paths["logs_dir"] / "stage3a_spatial_cv.log",
            skip_if=paths["spatial_csv"],
            force=args.force,
            dry_run=args.dry_run,
            logger=logger,
        )
        if ran:
            stages_run.append("spatial_cv")
    else:
        logger.info("[SKIP] Stage 3a (Spatial CV) — disabled via --skip-cv or config")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 3b: Temporal CV (forward chaining)
    # ════════════════════════════════════════════════════════════════════════
    if not skip_cv and run_temporal_cv:
        cmd_temporal_cv = [
            python, "cv_temporal_forward.py",
            "--config",      str(args.config),
            "--data-dir",    str(paths["data_dir"]),
            "--output-dir",  str(paths["temporal_cv_dir"]),
            "--lam",         str(best_lam),
            "--lam-t",       str(best_lam_t),
            "--sigma-insar", str(sigma_insar),
            "--sigma-well",  str(sigma_well),
        ]
        if no_plots:
            cmd_temporal_cv.append("--no-plots")

        ran = run_stage(
            name="Stage 3b: Temporal CV (forward chaining)",
            cmd=cmd_temporal_cv,
            log_path=paths["logs_dir"] / "stage3b_temporal_cv.log",
            skip_if=paths["temporal_csv"],
            force=args.force,
            dry_run=args.dry_run,
            logger=logger,
        )
        if ran:
            stages_run.append("temporal_cv")
    else:
        logger.info("[SKIP] Stage 3b (Temporal CV) — disabled via --skip-cv or config")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 4: GP Grid Expansion + Combined Uncertainty
    # ════════════════════════════════════════════════════════════════════════
    cmd_gp = [
        python, "stage2_gp_interpolation.py",
        "--inversion-npz", str(paths["inversion_npz"]),
        "--grid-nc",       str(paths["grid_nc"]),
        "--output-nc",     str(paths["gp_nc"]),
        "--n-restarts",    str(n_restarts),
        "--std-threshold", str(std_threshold),
        "--gp-mode",       gp_mode,
        "--angle-search",  angle_search,
        "--angle-min",     str(angle_min),
        "--angle-max",     str(angle_max),
    ]
    if max_anisotropy:
        cmd_gp += ["--max-anisotropy", max_anisotropy]
    # Add CV CSV paths only if both exist (or will exist after stages 3a/3b)
    spatial_csv_exists = paths["spatial_csv"].exists() or (
        not skip_cv and run_spatial_cv and not args.dry_run
    )
    temporal_csv_exists = paths["temporal_csv"].exists() or (
        not skip_cv and run_temporal_cv and not args.dry_run
    )
    if paths["spatial_csv"].exists():
        cmd_gp += ["--spatial-cv-csv", str(paths["spatial_csv"])]
    if paths["temporal_csv"].exists():
        cmd_gp += ["--temporal-cv-csv", str(paths["temporal_csv"])]

    # For dry-run, show what would be passed
    if args.dry_run and not skip_cv:
        cmd_gp += [
            "--spatial-cv-csv", str(paths["spatial_csv"]),
            "--temporal-cv-csv", str(paths["temporal_csv"]),
        ]

    ran = run_stage(
        name="Stage 4: GP Grid Expansion",
        cmd=cmd_gp,
        log_path=paths["logs_dir"] / "stage4_gp_expansion.log",
        skip_if=paths["gp_nc"],
        force=args.force,
        dry_run=args.dry_run,
        logger=logger,
    )
    if ran:
        stages_run.append("gp_expansion")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 5: Summary Report
    # ════════════════════════════════════════════════════════════════════════
    total_elapsed = time.perf_counter() - t_global_start

    if not args.dry_run:
        generate_summary(
            config_path=args.config,
            paths=paths,
            best_lam=best_lam,
            best_lam_t=best_lam_t,
            stages_run=stages_run,
            elapsed_s=total_elapsed,
            skip_cv=skip_cv,
            logger=logger,
        )
    else:
        print(f"\n  [DRY-RUN] Stage 5: Summary Report -> {paths['summary']}")

    # =========================================================================
    # Stage 6: Visualisations (optional)
    # =========================================================================
    if make_plots:
        vis_dir = paths["output_dir"] / "visualisations"
        cmd_vis = [
            python, "visualise_results.py",
            "--output-dir", str(paths["output_dir"]),
            "--vis-dir",    str(vis_dir),
        ]
        ran = run_stage(
            name="Stage 6: Visualisations",
            cmd=cmd_vis,
            log_path=paths["logs_dir"] / "stage6_visualisations.log",
            skip_if=vis_dir / "inversion" / "deep_residual_scatter.png",
            force=args.force,
            dry_run=args.dry_run,
            logger=logger,
        )
        if ran:
            stages_run.append("visualisations")

    if args.dry_run:
        print(f"\n=== End of dry run ===")


if __name__ == "__main__":
    main()
