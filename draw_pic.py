import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_across_methods(folders, noise_type):
    labels = [os.path.basename(folder.rstrip('/')).replace('_logs', '').capitalize() for folder in folders]
    colors = ["blue", "green", "orange", "red", "purple", "cyan", "gray"]  # 可扩展 , gray brown pink
    fig_qd, ax_qd = plt.subplots()
    fig_cov, ax_cov = plt.subplots()

    for folder, label, color in zip(folders, labels, colors):
        filename = f"{noise_type}_experiment_results.csv"
        file_path = os.path.join(folder, filename)

        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，跳过。")
            continue

        df = pd.read_csv(file_path)

        df_mean = df[df['Seed'] == 'Mean'].copy()
        df_std = df[df['Seed'] == 'STD'].copy()

        df_mean.loc[:, 'Parameter'] = df_mean['Parameter'].astype(float)
        df_std.loc[:, 'Parameter'] = df_std['Parameter'].astype(float)

        df_merged = pd.merge(df_mean, df_std, on='Parameter', suffixes=('_mean', '_std'))
        df_merged = df_merged.sort_values(by='Parameter')

        x = df_merged['Parameter']
        qd_mean = df_merged['QD Score_mean']
        qd_std = df_merged['QD Score_std']
        cov_mean = df_merged['Coverage_mean']
        cov_std = df_merged['Coverage_std']

        ax_qd.plot(x, qd_mean, label=label, color=color, marker='o', ms=3)
        ax_qd.fill_between(x, qd_mean - qd_std, qd_mean + qd_std, color=color, alpha=0.2, linewidth=0)

        ax_cov.plot(x, cov_mean, label=label, color=color, marker='o', ms=3)
        ax_cov.fill_between(x, cov_mean - cov_std, cov_mean + cov_std, color=color, alpha=0.2, linewidth=0)

    ax_qd.set_title(f"Impact of Noise Rate on QD Score ({noise_type.replace('_', ' ').title()})")
    ax_qd.set_xlabel("Noise Rate")
    ax_qd.set_ylabel("QD Score")
    ax_qd.legend()
    ax_qd.grid(True)

    ax_cov.set_title(f"Impact of Noise Rate on Coverage ({noise_type.replace('_', ' ').title()})")
    ax_cov.set_xlabel("Noise Rate")
    ax_cov.set_ylabel("Coverage")
    ax_cov.legend()
    ax_cov.grid(True)

    # 保存
    qd_path = f"arm/pic/qd_score_{noise_type}.png"
    cov_path = f"arm/pic/coverage_{noise_type}.png"
    fig_qd.savefig(qd_path, dpi=300, bbox_inches='tight')
    fig_cov.savefig(cov_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存为：\n- {qd_path}\n- {cov_path}")

# argparse 修改
parser = argparse.ArgumentParser(description="Compare robust methods under a specific noise type")
parser.add_argument('--noise_type', type=str, required=True, help="e.g. flip_labels_asymmetric")
parser.add_argument('--log_dirs', nargs='+', required=True, help="List of robust method folders")

args = parser.parse_args()
plot_across_methods(args.log_dirs, args.noise_type)

# python draw_pic.py   --noise_type noisy_labels_exact   --log_dirs /mnt/data6t/qdhf/arm/logs/reweight_logs /mnt/data6t/qdhf/arm/logs/rDPO_logs /mnt/data6t/qdhf/arm/logs/label_smoothing_logs /mnt/data6t/qdhf/arm/logs/raw_qdhf_logs /mnt/data6t/qdhf/arm/logs/cDPO_logs /mnt/data6t/qdhf/arm/logs/crdo_logs /mnt/data6t/qdhf/arm/logs/GAPO_logs
# python draw_pic.py   --noise_type flip_labels_asymmetric   --log_dirs /mnt/data6t/qdhf/arm/logs/reweight_logs /mnt/data6t/qdhf/arm/logs/rDPO_logs /mnt/data6t/qdhf/arm/logs/label_smoothing_logs /mnt/data6t/qdhf/arm/logs/raw_qdhf_logs /mnt/data6t/qdhf/arm/logs/cDPO_logs /mnt/data6t/qdhf/arm/logs/crdo_logs /mnt/data6t/qdhf/arm/logs/GAPO_logs
# python draw_pic.py   --noise_type stochastic   --log_dirs /mnt/data6t/qdhf/arm/logs/reweight_logs /mnt/data6t/qdhf/arm/logs/rDPO_logs /mnt/data6t/qdhf/arm/logs/label_smoothing_logs /mnt/data6t/qdhf/arm/logs/raw_qdhf_logs /mnt/data6t/qdhf/arm/logs/cDPO_logs /mnt/data6t/qdhf/arm/logs/crdo_logs /mnt/data6t/qdhf/arm/logs/GAPO_logs