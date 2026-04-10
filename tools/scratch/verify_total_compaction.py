import numpy as np
import pandas as pd
from pathlib import Path

def verify():
    # Paths
    gt_path = Path(r"D:\111_PROJECT_002\synthetic_challenge_data\ground_truth.npz")
    est_path = Path(r"D:\110_PROJECT_002\constrained_inversion_scripts\my_input_data_synthetic\output_015\real_m_est.npz")
    
    gt = np.load(gt_path)
    est = np.load(est_path)
    
    tw1, tw2, tw3 = gt['w1'], gt['w2'], gt['w3']
    
    # Inversion results
    m_est = est['m_est'] # (60, 12, 3)
    station_names = est['station_names']
    x_coords = est['x_twd97']
    y_coords = est['y_twd97']
    
    # Time definition from generation script
    n_months = 60
    t = np.arange(n_months)
    linear_trend = (20.0 / 60.0) * t
    seasonal = 5.0 * np.sin(2 * np.pi * t / 12.0)
    S_t = linear_trend + seasonal
    
    # End state (Month 60)
    S_end = S_t[-1] # ~20.0 mm
    
    # Grid coords
    x_1d = np.linspace(-10, 10, 20)
    y_1d = np.linspace(-10, 10, 20)
    
    print(f"Comparing Total Cumulative Compaction at Month 60:")
    
    total_true_all = []
    total_est_all = []
    
    for i in range(len(station_names)):
        x, y = x_coords[i], y_coords[i]
        ix = np.argmin(np.abs(x_1d - x))
        iy = np.argmin(np.abs(y_1d - y))
        
        # True total cumulative compaction per layer
        true_total = np.array([tw1[iy, ix], tw2[iy, ix], tw3[iy, ix]]) * S_end
        
        # Est total cumulative
        est_total = m_est[:, i, :].sum(axis=0)
        
        total_true_all.append(true_total)
        total_est_all.append(est_total)
        
        if i < 3:
            print(f"\nStation: {station_names[i]}")
            print(f"  True Total: {true_total}")
            print(f"  Est. Total: {est_total}")
            print(f"  Ratio (Est/True): {est_total / true_total}")

    total_true_all = np.array(total_true_all)
    total_est_all = np.array(total_est_all)
    
    mae = np.mean(np.abs(total_true_all - total_est_all))
    mean_true = np.mean(total_true_all)
    rel_err = mae / mean_true
    
    print(f"\nGlobal Comparison Summary:")
    print(f"  Mean True Compaction: {mean_true:.2f} mm")
    print(f"  Mean absolute Error:   {mae:.2f} mm")
    print(f"  Relative Error:       {rel_err*100:.1f} %")
    
    if rel_err < 0.20:
        print("\nSUCCESS: Total compaction magnitudes match well.")
    else:
        print("\nWARNING: Magnitudes are significantly different.")

if __name__ == "__main__":
    verify()
