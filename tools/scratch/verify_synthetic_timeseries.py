import numpy as np
import pandas as pd
from pathlib import Path

def verify():
    # Paths
    gt_path = Path(r"D:\111_PROJECT_002\synthetic_challenge_data\ground_truth.npz")
    est_path = Path(r"D:\110_PROJECT_002\constrained_inversion_scripts\my_input_data_synthetic\output_015\real_m_est.npz")
    
    gt = np.load(gt_path)
    est = np.load(est_path)
    
    # Ground Truth weights
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
    dS_t = np.diff(S_t, prepend=0)
    
    # Grid coords
    x_1d = np.linspace(-10, 10, 20)
    y_1d = np.linspace(-10, 10, 20)
    
    print(f"Comparing Time Series for {len(station_names)} stations...")
    
    all_corrs = []
    all_rmses = []
    
    for i in range(len(station_names)):
        x, y = x_coords[i], y_coords[i]
        ix = np.argmin(np.abs(x_1d - x))
        iy = np.argmin(np.abs(y_1d - y))
        
        # True incremental compaction (note: in generation script it is -w * S_t, 
        # but the inversion recovers POSITIVE compaction values)
        # S_t in script is used as - (w * S_t)
        # So true positive compaction is w * dS_t
        true_m = np.zeros((n_months, 3))
        true_m[:, 0] = tw1[iy, ix] * dS_t
        true_m[:, 1] = tw2[iy, ix] * dS_t
        true_m[:, 2] = tw3[iy, ix] * dS_t
        
        # Est m
        e_m = m_est[:, i, :]
        
        # Stats
        for layer in range(3):
            corr = np.corrcoef(true_m[:, layer], e_m[:, layer])[0, 1]
            rmse = np.sqrt(np.mean((true_m[:, layer] - e_m[:, layer])**2))
            all_corrs.append(corr)
            all_rmses.append(rmse)
            
    print(f"\nTime Series Comparison Results:")
    print(f"  Mean Correlation: {np.mean(all_corrs):.4f}")
    print(f"  Mean RMSE (mm):   {np.mean(all_rmses):.4f}")
    
    if np.mean(all_corrs) > 0.95:
        print("\nSUCCESS: Time series match is excellent!")
    else:
        print("\nWARNING: Correlation is lower than expected.")

if __name__ == "__main__":
    verify()
