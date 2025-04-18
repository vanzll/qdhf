import subprocess
import pandas as pd
import numpy as np
import re
import os  # 用于检查文件是否存在

# 实验设置
noisy_methods = {
    # "noisy_labels_exact": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "stochastic": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # "add_equal_noise": [1, 2, 5, 10, 15, 20, 25, 30],
    # "flip_by_distance": [1, 2, 5, 10, 15, 20, 25, 30],
    # "flip_labels_asymmetric": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}
base_trial_ids = {
    "stochastic": 10,
    "add_equal_noise": 20,
    "flip_by_distance": 30,
    "flip_labels_asymmetric": 43,
    "noisy_labels_exact": 50
}
seeds = ["1111", "2222", "3333", "4444"]

# 主逻辑
for method, params in noisy_methods.items():
    # 为每种噪声方法生成不同的 CSV 文件路径
    csv_path = f"/mnt/nvme3n1/qdhf/maze/{method}_experiment_results.csv"

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
            print(f"Running experiment with method={method}, param={param}, trial_id={trial_id}, seed={seed}")
            # 实际运行调用（如果要跑真实代码，可以取消注释）
            result = subprocess.run(
                ["python", "main.py", "--seed", seed, "--trial_id", str(trial_id),
                 "--noisy_method", method, "--parameter", str(param)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            print(output)

            # 提取 QD score 和 Coverage（从最后一行提取）
            try:
                # 使用正则表达式匹配 QD score 和 Coverage
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
            # 如果是第一次写入，写表头，否则追加
            pd.DataFrame([record]).to_csv(csv_path, mode='a', header=header_written, index=False)
            header_written = False  # 只有第一次写入时才写表头

        # 可选：如果需要添加统计数据（均值和标准差），可以在此进行计算
        qd_mean = np.nanmean(qd_scores)
        qd_std = np.nanstd(qd_scores)
        coverage_mean = np.nanmean(coverage_scores)
        coverage_std = np.nanstd(coverage_scores)

        # 将统计数据写入 CSV
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
