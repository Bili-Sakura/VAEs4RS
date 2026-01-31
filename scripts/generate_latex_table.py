"""
Generate LaTeX tables from experiment results.

Outputs formatted tables ready for the paper.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_results(results_path: str) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_main_table(results: dict) -> str:
    """Generate LaTeX table for main results."""
    latex = r"""
\begin{table}[h]
\caption{Main Results: Comparison of VAEs on Remote Sensing Reconstruction.}
\label{tab:main_results}
\begin{center}
\begin{tabular}{lccccc}
\bf Dataset & \bf Model & \bf PSNR $\uparrow$ & \bf SSIM $\uparrow$ & \bf LPIPS $\downarrow$ & \bf FID $\downarrow$ \\
\hline \\
"""
    
    for model_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            if metrics is None:
                continue
            psnr = metrics.get('psnr', 0)
            ssim = metrics.get('ssim', 0)
            lpips = metrics.get('lpips', 0)
            fid = metrics.get('fid', 0)
            
            latex += f"{dataset_name} & {model_name} & {psnr:.2f} & {ssim:.4f} & {lpips:.4f} & {fid:.2f} \\\\\n"
        latex += "\\hline\n"
    
    latex += r"""
\end{tabular}
\end{center}
\end{table}
"""
    return latex


def generate_ablation_table(results: dict) -> str:
    """Generate LaTeX table for ablation study results."""
    latex = r"""
\begin{table}[h]
\caption{Ablation Study: VAE as Pre-processor for Denoising/De-hazing.}
\label{tab:ablation}
\begin{center}
\begin{tabular}{lcccc}
\bf Model & \bf Distortion & \bf Distorted PSNR & \bf Cleaned PSNR & \bf Improvement \\
\hline \\
"""
    
    for dataset_name, models in results.items():
        latex += f"\\multicolumn{{5}}{{c}}{{\\bf {dataset_name}}} \\\\\n\\hline\n"
        for model_name, distortions in models.items():
            for dist_name, metrics in distortions.items():
                dist_psnr = metrics['distorted']['psnr']
                clean_psnr = metrics['cleaned']['psnr']
                improvement = metrics['improvement']['psnr']
                sign = "+" if improvement > 0 else ""
                
                latex += f"{model_name} & {dist_name} & {dist_psnr:.2f} & {clean_psnr:.2f} & {sign}{improvement:.2f} \\\\\n"
        latex += "\\hline\n"
    
    latex += r"""
\end{tabular}
\end{center}
\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--type", type=str, default="main", choices=["main", "ablation"], help="Table type")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()
    
    results = load_results(args.results)
    
    if args.type == "main":
        latex = generate_main_table(results)
    else:
        latex = generate_ablation_table(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex)
        print(f"Saved to {args.output}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
