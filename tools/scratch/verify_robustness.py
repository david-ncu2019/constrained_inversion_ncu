"""
verify_robustness.py
====================
Compares inversion output against ground truth for all stress-test cases
(2a, 2b, 3a, 3b, 4) and prints a PASS/FAIL table with per-case RMSE thresholds.

Usage
-----
  uv run python tools/scratch/verify_robustness.py             # all cases
  uv run python tools/scratch/verify_robustness.py --case case2a
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\111_PROJECT_002\synthetic_robustness_data")
PIPELINE_OUT = "output"

CASES = {
    # case_id : (output_subdir_relative_to_case_dir, n_layers, rmse_threshold)
    "case2a": ("case2a_sparsity_12",  3, 0.05),
    "case2b": ("case2b_sparsity_6",   3, 0.15),
    "case3a": ("case3a_noise_3mm",    3, 0.08),
    "case3b": ("case3b_noise_5mm",    3, 0.12),
    "case4":  ("case4_complex",       5, 0.12),
}

# Grid coordinates (must match generator — 20x20 over [-10, 10])
x_1d = np.linspace(-10, 10, 20)
y_1d = np.linspace(-10, 10, 20)


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def verify_case(case_id: str) -> dict:
    """
    Load ground truth and inversion estimate, compute per-layer RMSE
    over all stations and all epochs.

    Returns a dict with keys: case_id, n_layers, rmse_threshold,
    layer_rmse (list), overall_rmse, status ('PASS'/'FAIL'/'MISSING').
    """
    case_dir_name, n_layers, threshold = CASES[case_id]
    case_dir  = BASE_DIR / case_dir_name
    gt_path   = case_dir / "ground_truth.npz"
    est_path  = case_dir / PIPELINE_OUT / "real_m_est.npz"

    result = {
        "case_id":      case_id,
        "n_layers":     n_layers,
        "threshold":    threshold,
        "layer_rmse":   None,
        "overall_rmse": None,
        "status":       "MISSING",
        "detail":       "",
    }

    if not gt_path.exists():
        result["detail"] = f"Ground truth missing: {gt_path}"
        return result

    if not est_path.exists():
        result["detail"] = f"Inversion output missing: {est_path}"
        return result

    gt  = np.load(gt_path)
    est = np.load(est_path)

    # --- Inversion output: shape (n_epochs, n_stations, n_layers)
    m_est    = est["m_est"]                  # (T, S, L)
    x_coords = est["x_twd97"]               # (S,)
    y_coords = est["y_twd97"]               # (S,)
    n_epochs, n_stations, n_layers_est = m_est.shape

    if n_layers_est != n_layers:
        result["detail"] = (
            f"Layer count mismatch: expected {n_layers}, got {n_layers_est}"
        )
        result["status"] = "FAIL"
        return result

    # --- Temporal signal from ground truth
    S_t = gt["S_t"]          # (T,)  cumulative

    # --- Compute RMSE per layer
    layer_errors = []
    for l in range(n_layers):
        key = f"w{l + 1}"
        w_field = gt[key]                    # (20, 20)

        station_errors = []
        for s_idx in range(n_stations):
            ix = int(np.argmin(np.abs(x_1d - x_coords[s_idx])))
            iy = int(np.argmin(np.abs(y_1d - y_coords[s_idx])))
            gt_cum  = w_field[iy, ix] * S_t          # ground truth (T,)
            est_cum = m_est[:, s_idx, l]             # estimate     (T,)
            rmse    = float(np.sqrt(np.mean((gt_cum - est_cum) ** 2)))
            station_errors.append(rmse)

        layer_errors.append(float(np.mean(station_errors)))

    overall = float(np.mean(layer_errors))
    status  = "PASS" if overall <= threshold else "FAIL"

    result.update(
        layer_rmse=layer_errors,
        overall_rmse=overall,
        status=status,
        detail="",
    )
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _layer_rmse_str(layer_rmse: list[float] | None, n_layers: int) -> str:
    if layer_rmse is None:
        return " — " * n_layers
    return "  |  ".join(f"{v:.4f}" for v in layer_rmse)


def print_report(results: list[dict]) -> None:
    """Print structured PASS/FAIL table to stdout."""
    header_cols = ["Case", "Status", "Threshold", "Overall RMSE", "Layer RMSE (per layer)", "Detail"]
    rows = []

    for r in results:
        layer_str = _layer_rmse_str(r["layer_rmse"], r["n_layers"])
        rows.append([
            r["case_id"],
            r["status"],
            f"{r['threshold']:.2f}",
            f"{r['overall_rmse']:.4f}" if r["overall_rmse"] is not None else "—",
            layer_str,
            r["detail"],
        ])

    # Column widths
    col_widths = [max(len(header_cols[j]), max(len(row[j]) for row in rows)) for j in range(len(header_cols))]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    print("\n")
    print("=" * 70)
    print("  PIPELINE ROBUSTNESS VERIFICATION REPORT")
    print("=" * 70)
    print(sep)
    print(fmt.format(*header_cols))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)

    # Summary
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_miss = sum(1 for r in results if r["status"] == "MISSING")
    print(f"\n  PASS: {n_pass}   FAIL: {n_fail}   MISSING: {n_miss}   "
          f"(Total: {len(results)})")

    if n_fail > 0 or n_miss > 0:
        print("\n  [!] Some cases did not pass. Review the detail column above.")
    else:
        print("\n  [OK] All cases PASSED. Pipeline is robust.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify inversion output against synthetic ground truth."
    )
    parser.add_argument(
        "--case",
        choices=list(CASES.keys()),
        default=None,
        help="Run verification for a single case (default: all).",
    )
    args = parser.parse_args()

    case_ids = [args.case] if args.case else list(CASES.keys())
    results = [verify_case(cid) for cid in case_ids]
    print_report(results)


if __name__ == "__main__":
    main()
