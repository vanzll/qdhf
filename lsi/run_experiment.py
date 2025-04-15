import subprocess
import os

# 定义 noisy_labels_exact 和 seeds
noisy_labels_exact = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seeds = ["1111", "2222", "3333", "4444"]

# 遍历每个 noisy_labels_exact 的参数和每个 seed
for parameter in noisy_labels_exact:
    for seed in seeds:
        # 构建命令
        command = f"python main.py --noisy_method noisy_labels_exact --parameter {parameter} --seed {seed}"
        
        # 打印并执行命令
        print(f"Running: {command}")
        subprocess.run(command, shell=True)
