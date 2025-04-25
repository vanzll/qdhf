import subprocess
import pandas as pd
import numpy as np
import re
import os  
import gc
import time
import torch
import argparse

parser = argparse.ArgumentParser(description="Run QDHF experiments with config")
parser.add_argument("--noise_type", type=str, required=True)
parser.add_argument("--robust", type=str)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
args = parser.parse_args()

# 实验设置
noisy_list = {
    "noisy_labels_exact": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], # 0.05, 0.1, 0.2, 
    "stochastic": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "add_equal_noise": [1, 2, 5, 10, 15, 20, 25, 30],
    "flip_by_distance": [1, 2, 5, 10, 15, 20, 25, 30],
    "flip_labels_asymmetric": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}
noisy_methods = {args.noise_type: noisy_list[args.noise_type]}
base_trial_ids = {
    "stochastic": 10,
    "add_equal_noise": 20,
    "flip_by_distance": 30,
    "flip_labels_asymmetric": 40,
    "noisy_labels_exact": 50
}
seeds = ["1111", "2222", "3333", "4444"]

# 主逻辑
for method, params in noisy_methods.items():
    csv_path = f"/mnt/data6t/qdhf/lsi/logs/{args.robust}_logs/{method}_experiment_results.csv"

    # 检查文件是否存在，如果存在则跳过创建
    if os.path.exists(csv_path):
        print(f"CSV file for method {method} already exists, skipping creation.")
        header_written = False
    else:
        header_written = True

    base_trial_id = base_trial_ids[method]
    for i, param in enumerate(params):
        trial_id = base_trial_id + i

        qd_scores = []
        coverage_scores = []
        

        for seed in seeds:
            print(f"Running experiment with method={method}, param={param}, robust={args.robust}, trial_id={trial_id}, seed={seed}")
            result = subprocess.run(
                ["python", "main.py", "--seed", seed, "--trial_id", str(trial_id),
                 "--noisy_method", method, "--parameter", str(param), "--robust_loss", 
                 args.robust, "--device", args.device],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            result.check_returncode()
            output = result.stdout
            print(output)

            try:
                match = re.search(r"QD score: ([\d\.]+) Coverage: ([\d\.]+)", output)
                if match:
                    qd_score = float(match.group(1))
                    coverage = float(match.group(2))
                else:
                    qd_score, coverage = np.nan, np.nan
            except Exception:
                qd_score, coverage = np.nan, np.nan

            qd_scores.append(qd_score)
            coverage_scores.append(coverage)

            # 每次实验结果写入 CSV
            record = {
                "Method": method,
                "Parameter": param,
                "Seed": seed,
                "Trial ID": trial_id,
                "QD Score": qd_score,
                "Coverage": coverage
            }

            pd.DataFrame([record]).to_csv(csv_path, mode='a', header=header_written, index=False)
            header_written = False  # 只有第一次写入时才写表头
            
            gc.collect()
            time.sleep(2)

        qd_mean = np.nanmean(qd_scores)
        qd_std = np.nanstd(qd_scores)
        coverage_mean = np.nanmean(coverage_scores)
        coverage_std = np.nanstd(coverage_scores)

        stats_record = {
            "Method": method,
            "Parameter": param,
            "Seed": "Mean",
            "Trial ID": "Mean",
            "QD Score": qd_mean,
            "Coverage": coverage_mean
        }
        pd.DataFrame([stats_record]).to_csv(csv_path, mode='a', header=False, index=False)

        stats_record = {
            "Method": method,
            "Parameter": param,
            "Seed": "STD",
            "Trial ID": "STD",
            "QD Score": qd_std,
            "Coverage": coverage_std
        }
        pd.DataFrame([stats_record]).to_csv(csv_path, mode='a', header=False, index=False)

print(f"Experiment results have been saved to {csv_path}")
