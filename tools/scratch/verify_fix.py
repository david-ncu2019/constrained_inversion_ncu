import numpy as np
from pathlib import Path

def verify():
    gt_path = Path(r"D:\111_PROJECT_002\synthetic_challenge_data\ground_truth.npz")
    est_path = Path(r"D:\110_PROJECT_002\constrained_inversion_scripts\my_input_data_synthetic\output_015\real_m_est.npz")
    
    gt = np.load(gt_path)
    est = np.load(est_path)
    
    tw1, tw2, tw3, tD = gt['w1'], gt['w2'], gt['w3'], gt['D']
    m_est = est['m_est'] # (60, 12, 3) - these are CURRENTLY CUMULATIVE in this run
    x_coords = est['x_twd97']
    y_coords = est['y_twd97']
    
    n_months = 60
    t = np.arange(n_months)
    S_t = (20.0 / 60.0) * t + 5.0 * np.sin(2 * np.pi * t / 12.0)
    
    x_1d = np.linspace(-10, 10, 20)
    y_1d = np.linspace(-10, 10, 20)
    
    print("Inversion Quality Assessment:")
    print("----------------------------")
    
    layer_errors = []
    
    for i in range(len(x_coords)):
        ix = np.argmin(np.abs(x_1d - x_coords[i]))
        iy = np.argmin(np.abs(y_1d - y_coords[i]))
        
        # Ground Truth Cumulative Compaction
        # Compaction = weight * S(t)
        gt_cum = np.zeros((n_months, 3))
        gt_cum[:, 0] = tw1[iy, ix] * S_t
        gt_cum[:, 1] = tw2[iy, ix] * S_t
        gt_cum[:, 2] = tw3[iy, ix] * S_t
        
        # Estimate
        e_cum = m_est[:, i, :]
        
        # Calculate RMSE per layer over all time steps
        rmse = np.sqrt(np.mean((gt_cum - e_cum)**2, axis=0))
        layer_errors.append(rmse)
        
    avg_rmse = np.mean(layer_errors, axis=0) # (3,)
    
    print(f"Layer-wise RMSE (cumulative, over 60 months):")
    for l, val in enumerate(avg_rmse):
        print(f"  Layer {l+1} ({[10, 50, 100][l]}m): {val:.4f} mm")
        
    overall_rmse = np.mean(avg_rmse)
    print(f"\nOverall Mean RMSE: {overall_rmse:.4f} mm")
    
    # Check Max values
    gt_max = np.mean(tw1+tw2+tw3) * 20 # Avg max compaction ~ 0.9 * 20 = 18
    print(f"Typical Full Compaction Magnitude: ~{gt_max:.1f} mm")
    
    if overall_rmse < 1.0:
        print("\n✅ EXCELLENT FIT: Errors are less than 1mm on average.")
    elif overall_rmse < 3.0:
        print("\n🟡 GOOD FIT: Some discrepancies but trends captured.")
    else:
        print("\n❌ POOR FIT: Significant deviation from ground truth.")

if __name__ == "__main__":
    verify()
