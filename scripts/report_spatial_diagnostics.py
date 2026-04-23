import os
from pathlib import Path
import glob

def generate_report(diag_dir="output/diagnostics_spatial", output_file="output/spatial_diagnostic_report.md"):
    diag_path = Path(diag_dir)
    if not diag_path.exists():
        print(f"Directory {diag_dir} not found. Run the pipeline first.")
        return

    # Find all param files
    param_files = sorted(glob.glob(str(diag_path / "layer_*_params.txt")), 
                         key=lambda x: int(os.path.basename(x).split('_')[1]))
    
    report_lines = [
        "# Spatial Interpolation Diagnostic Report",
        "",
        "This report summarizes the variogram models and parameters selected via Optuna optimization for each depth layer.",
        "",
        "## Summary Table",
        "",
        "| Layer | Best Model | Angle (deg) | Ratio | sill | range | nugget |",
        "|-------|------------|-------------|-------|------|-------|--------|"
    ]
    
    for pf in param_files:
        layer_idx = os.path.basename(pf).split('_')[1]
        params = {}
        with open(pf, 'r') as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    params[k.strip()] = v.strip()
        
        report_lines.append(
            f"| {layer_idx} | {params.get('Best Model', 'N/A')} | "
            f"{float(params.get('angle', 0)):.2f} | "
            f"{float(params.get('scaling', 1)):.2f} | "
            f"{float(params.get('psill', 0)):.4f} | "
            f"{float(params.get('range', 0)):.2f} | "
            f"{float(params.get('nugget', 0)):.4f} |"
        )
    
    report_lines.append("\n## Variograms per Layer\n")
    
    for pf in param_files:
        layer_idx = os.path.basename(pf).split('_')[1]
        img_name = f"layer_{layer_idx}_variogram.png"
        img_path = diag_path / img_name
        
        if img_path.exists():
            report_lines.append(f"### Layer {layer_idx}")
            report_lines.append(f"![Variogram Layer {layer_idx}]({img_path.absolute().as_uri()})")
            report_lines.append("")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    generate_report()
