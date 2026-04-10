import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def verify():
    # Paths
    gt_path = Path(r"D:\111_PROJECT_002\synthetic_challenge_data\ground_truth.npz")
    est_path = Path(r"D:\110_PROJECT_002\constrained_inversion_scripts\my_input_data_synthetic\output_015\real_m_est.npz")
    
    if not gt_path.exists():
        print(f"❌ Ground truth not found at {gt_path}")
        return
    if not est_path.exists():
        print(f"❌ Estimate not found at {est_path}")
        return
        
    # Load data
    gt = np.load(gt_path)
    est = np.load(est_path)
    
    # Ground Truth arrays (20, 20)
    tw1, tw2, tw3, tD = gt['w1'], gt['w2'], gt['w3'], gt['D']
    
    # Estimated data
    # depth_weights shape: (n_stations, n_layers)
    e_weights = est['depth_weights']
    station_names = est['station_names']
    x_coords = est['x_twd97']
    y_coords = est['y_twd97']
    
    # Grid coords from synthetic script
    x_1d = np.linspace(-10, 10, 20)
    y_1d = np.linspace(-10, 10, 20)
    
    print(f"Checking {len(station_names)} stations...")
    
    diffs = []
    for i, name in enumerate(station_names):
        x, y = x_coords[i], y_coords[i]
        
        # Find nearest grid indices
        ix = np.argmin(np.abs(x_1d - x))
        iy = np.argmin(np.abs(y_1d - y))
        
        # True fractional weights
        # In synthetic: InSAR_cum = (w1+w2+w3) * S_t + D
        # So "weight" of layer 1 is w1 * S_t / InSAR_cum
        # Over long term, weight -> w1 / (w1+w2+w3 + D/S_t)
        # But D is constant, S_t grows.
        # Actually, the inversion usually defines weight as fraction of InSAR *compaction*
        # If D is constant, dD/dt = 0. So incremental InSAR = (w1+w2+w3) * dS/dt.
        # Thus incremental weights should exactly match w1 / (w1+w2+w3) ??? 
        # Wait, if D is constant, it doesn't show up in incremental InSAR!
        
        # Let's check w1, w2, w3 directly.
        # In the inversion, m_est for layer 1 is w1 * dS.
        # InSAR incremental is sum(wi)*dS.
        # So m_est / InSAR_inc = w1 / sum(wi).
        
        denominator = tw1[iy, ix] + tw2[iy, ix] + tw3[iy, ix]
        # Wait, if there's deep residual D, InSAR_inc is still just sum(wi)*dS?
        # Yes, because D is constant in time.
        
        # Actually, the user rules say: "G @ m calculates only the partial surface displacement from the shallow 0-300m layers."
        # The sum of m should equal the InSAR if D=0. If D>0, sum(m) < InSAR?
        # No, if D is constant, InSAR_incremental = sum(m_layers).
        
        w_true = np.array([tw1[iy, ix], tw2[iy, ix], tw3[iy, ix]]) / denominator
        w_est = e_weights[i]
        
        error = np.abs(w_true - w_est)
        diffs.append(error)
        
        if i < 3: # Print first few
            print(f"\nStation: {name} (x={x:.2f}, y={y:.2f})")
            print(f"  True Weights (w/sum): {w_true}")
            print(f"  Est. Weights:        {w_est}")
            print(f"  Abs Error:           {error}")

    avg_error = np.mean(diffs, axis=0)
    print(f"\nAverage Weight Error per layer: {avg_error}")
    print(f"Overall Mean Absolute Error: {np.mean(avg_error):.6f}")

    if np.all(avg_error < 0.01):
        print("\n✅ MATCH DETECTED! Results are very close to ground truth.")
    else:
        print("\n⚠️ DISCREPANCY FOUND. Significant error in weights.")

if __name__ == "__main__":
    verify()
