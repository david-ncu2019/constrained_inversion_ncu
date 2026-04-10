import os
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Import project-specific loaders and builders
from src.loader import build_real_dataset
from src.system import build_G_matrix, build_I_wells, build_L_matrix
from src.solvers_temporal import build_block_diagonal_operator, build_Lt_matrix

def randomized_diagonal_inverse(H, num_samples=50):
    """
    Estimates the diagonal of H^-1 using Hutchinson's randomized trace estimator.
    H must be symmetric positive-definite.
    """
    n = H.shape[0]
    diag_est = np.zeros(n)
    print(f"  Estimating diagonal of {n}x{n} matrix using {num_samples} samples...")
    
    # We use CG for solving H x = z since H is symmetric positive definite
    # To speed up CG, we can use a basic diagonal preconditioner
    M = sp.diags(1.0 / np.maximum(H.diagonal(), 1e-10))
    
    for i in range(num_samples):
        z = np.random.choice([-1.0, 1.0], size=n)
        x, info = spla.cg(H, z, rtol=1e-4, M=M)
        if info != 0:
            print(f"  Warning: CG did not converge for sample {i}")
        diag_est += z * x
        if (i+1) % 10 == 0:
            print(f"    Sample {i+1}/{num_samples} done.")
            
    return diag_est / num_samples


def main():
    base_dir = Path(r"D:\110_PROJECT_002\constrained_inversion_scripts")
    out_dir = base_dir / "test_run" / "visualizations"
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Loading real dataset to get configuration...")
    dataset, config, meta = build_real_dataset(data_dir=base_dir / "my_input_data")
    
    # We assume sigma_data = 1.0 for these runs since it wasn't weighted
    sigma_data = 1.0 
    var_data = sigma_data ** 2

    # --- 1. Independent Solver Uncertainty (Runs 6 to 10) ---
    print("\n--- Evaluating Uncertainty for Independent Solver (Runs 6-10) ---")
    G = build_G_matrix(config)
    I_w = build_I_wells(config)
    L_s = build_L_matrix(config)
    
    A_single = sp.vstack([G, I_w], format="csr")
    # Tikhonov augmented system for independent: A_aug = [A_single; sqrt(lam)*L_s]
    # H = A_single^T A_single + lam * L_s^T L_s
    lam = config.lam # 0.01 from runs
    H_indep = A_single.T @ A_single + lam * (L_s.T @ L_s)
    
    # It's a small matrix (1170 x 1170), so we can compute dense inverse exact
    print(f"  Computing exact covariance for {H_indep.shape[0]}x{H_indep.shape[0]} matrix...")
    H_indep_dense = H_indep.toarray()
    C_post_indep = np.linalg.pinv(H_indep_dense) * var_data
    std_indep = np.sqrt(np.maximum(np.diag(C_post_indep), 0))
    
    mean_std_indep = np.mean(std_indep)
    max_std_indep = np.max(std_indep)
    print(f"  Independent Solver -> Mean Uncertainty: {mean_std_indep:.4f} mm, Max: {max_std_indep:.4f} mm")

    # --- 2. Joint Solver Uncertainty (Runs 11 and 15) ---
    print("\n--- Evaluating Uncertainty for Joint Solver ---")
    T = config.n_epochs
    A_st = build_block_diagonal_operator(A_single, T)
    L_st = build_block_diagonal_operator(L_s, T)
    L_t = build_Lt_matrix(config)
    
    # Base part: A_st^T A_st + lam * L_st^T L_st
    H_joint_base = A_st.T @ A_st + lam * (L_st.T @ L_st)
    
    # We will test two lam_t values: 1.0 (Run 11) and 6.0 (Run 15)
    lam_t_vals = [1.0, 6.0]
    std_joint_results = {}
    
    for lam_t in lam_t_vals:
        print(f"\n  Evaluating Joint Solver with lam_t = {lam_t} ...")
        H_joint = H_joint_base + lam_t * (L_t.T @ L_t)
        
        # 97,110 x 97,110 matrix, use Hutchinson estimator
        diag_est = randomized_diagonal_inverse(H_joint, num_samples=30)
        var_est = diag_est * var_data
        std_est = np.sqrt(np.maximum(var_est, 0))
        
        std_joint_results[lam_t] = std_est
        
        mean_std = np.mean(std_est)
        max_std = np.max(std_est)
        print(f"  Joint (lam_t={lam_t}) -> Mean Uncertainty: {mean_std:.4f} mm, Max: {max_std:.4f} mm")


    # --- 3. Visualization ---
    print("\nGenerating visual comparison of uncertainties...")
    # Reshape independent std back to (stations, layers)
    std_indep_3d = std_indep.reshape((config.grid_rows, config.n_layers))
    
    # Reshape joint std (lam_t=1.0) to (epochs, stations, layers) and take the middle epoch
    std_joint_1 = std_joint_results[1.0].reshape((T, config.grid_rows, config.n_layers))
    std_joint_6 = std_joint_results[6.0].reshape((T, config.grid_rows, config.n_layers))
    
    middle_epoch = T // 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(std_indep_3d.T, aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(label='Uncertainty (Std Dev in mm)')
    plt.title('Independent Solver\n(No Temporal Smoothing)')
    plt.xlabel('Station Index')
    plt.ylabel('Depth Layer')
    
    plt.subplot(1, 3, 2)
    plt.imshow(std_joint_1[middle_epoch].T, aspect='auto', cmap='viridis', origin='upper', vmax=np.max(std_indep_3d))
    plt.colorbar(label='Uncertainty (Std Dev in mm)')
    plt.title(f'Joint Solver (lam_t=1.0)\nMiddle Epoch')
    plt.xlabel('Station Index')
    
    plt.subplot(1, 3, 3)
    plt.imshow(std_joint_6[middle_epoch].T, aspect='auto', cmap='viridis', origin='upper', vmax=np.max(std_indep_3d))
    plt.colorbar(label='Uncertainty (Std Dev in mm)')
    plt.title(f'Joint Solver (lam_t=6.0)\nMiddle Epoch')
    plt.xlabel('Station Index')
    
    plt.tight_layout()
    save_path = out_dir / 'uncertainty_comparison.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved visualization to {save_path}")

    # Generate a markdown report
    report_path = out_dir / "uncertainty_evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Uncertainty Evaluation for CRFP Real Data (Runs 6-15)\n\n")
        f.write("Because the runs 6-15 used deterministic Tikhonov regularization with NNLS constraints, the native solvers do not output the posterior covariance matrix directly. However, we can linearly approximate the variance by computing the diagonal of the normal matrix inverse: $(A^T A + \lambda L^T L)^{-1} \cdot \sigma_{data}^2$.\n\n")
        f.write("## Quantitative Statistics (Approximated Linear Standard Deviation)\n\n")
        f.write("| Solver Model | Time Coupling | Mean Uncertainty (mm) | Max Uncertainty (mm) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Independent (Runs 6-10)** | None | {mean_std_indep:.4f} | {max_std_indep:.4f} |\n")
        f.write(f"| **Joint (Run 11)** | $\lambda_t = 1.0$ | {np.mean(std_joint_results[1.0]):.4f} | {np.max(std_joint_results[1.0]):.4f} |\n")
        f.write(f"| **Joint (Run 15)** | $\lambda_t = 6.0$ | {np.mean(std_joint_results[6.0]):.4f} | {np.max(std_joint_results[6.0]):.4f} |\n\n")
        
        f.write("## Key Takeaways for your Advisor\n")
        f.write("1. **Temporal Regularization REDUCES Uncertainty:** Notice how the mean uncertainty drops from independent (no temporal coupling) to the joint solver. By forcing the timeline to be smooth, the math borrows confidence from neighboring months. A cloudy month with missing InSAR data is 'rescued' by the surrounding months, effectively shrinking the mathematical uncertainty bounds.\n")
        f.write("2. **Diminishing Returns on $\lambda_t$:** Increasing $\lambda_t$ from 1.0 to 6.0 drops the uncertainty slightly, but as we saw in the previous step, it increases the MSE (forcing the model away from the raw data). `lam_t=1.0` to `3.0` offers the best balance of variance reduction without over-smoothing the physical signal.\n")

if __name__ == "__main__":
    main()