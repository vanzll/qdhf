#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python run_experiment.py
# python main.py --seed 2222 --trial_id 52 --noisy_method noisy_labels_exact --parameter 0.2
# 设置噪声方法和对应的参数列表
# declare -A noisy_methods
# noisy_methods["stochastic"]='0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9'
# noisy_methods["add_equal_noise"]='1 2 5 10'
# noisy_methods["flip_by_distance"]='1 2 5 10 15 20 25 30'
# noisy_methods["flip_labels_asymmetric"]='0.025 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45'
# noisy_methods["noisy_labels_exact"]='0.05 0.1 0.2'

# base_trial_id_stochastic=10
# base_trial_id_equal_noise=20
# base_trial_id_flip_by_distance=30
# base_trial_id_flip_labels_asymmetric=40
# base_trial_id_noisy_labels_exact=50

# seeds=("1111" "2222" "3333" "4444")

# declare -A qd_scores
# declare -A coverage_scores

# output_csv="experiment_results.csv"
# echo "Method,Parameter,Seed,Trial ID,QD Score,Coverage" > $output_csv

# for method in "${!noisy_methods[@]}"; do
#     params=(${noisy_methods[$method]})

#     if [ "$method" == "stochastic" ]; then
#         trial_id=$base_trial_id_stochastic
#     elif [ "$method" == "add_equal_noise" ]; then
#         trial_id=$base_trial_id_equal_noise
#     elif [[ "$method" == "flip_by_distance" ]]; then
#         trial_id=$base_trial_id_flip_by_distance
#     elif [[ "$method" == "flip_labels_asymmetric" ]]; then
#         trial_id=$base_trial_id_flip_labels_asymmetric
#     elif [[ "$method" == "noisy_labels_exact" ]]; then
#         trial_id=$base_trial_id_noisy_labels_exact
#     else
#         trial_id=0
#     fi

#     # 遍历每个参数，计算并设置 trial_id
#     for param in "${params[@]}"; do
#         # 存储当前 method 和 param 的 qd score 和 coverage 的数组
#         qd_scores=()
#         coverage_scores=()

#         for seed in "${seeds[@]}"; do
#             # 执行 Python 脚本，传入 trial_id、noisy_method、parameter 和 seed
#             echo "Running experiment with method: $method, parameter: $param, trial_id: $trial_id, seed: $seed"
            
#             # 捕获输出的 qd score 和 coverage
#             output=$(python main.py --seed $seed --trial_id $trial_id --noisy_method $method --parameter $param)
            
#             # 提取 qd score 和 coverage，假设它们在输出中是以类似 "QD score: X" 和 "Coverage: Y" 的格式出现
#             qd_score=$(echo "$output" | grep -oP 'QD score: \K[0-9.]+')
#             coverage=$(echo "$output" | grep -oP 'Coverage: \K[0-9.]+')
            
#             echo "$method,$param,$seed,$trial_id,$qd_score,$coverage" >> $output_csv

#             # 将 qd score 和 coverage 添加到数组中
#             qd_scores+=("$qd_score")
#             coverage_scores+=("$coverage")


#             # 增加 trial_id，以便下一个实验使用新的 ID
#             ((trial_id++))
#         done

#         # 计算 qd score 和 coverage 的 mean 和 std
#         qd_mean=$(echo "${qd_scores[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i} END {print s/NF}')
#         qd_std=$(echo "${qd_scores[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i; sumsq+=$i*$i} END {print sqrt(sumsq/NF - (s/NF)^2)}')

#         coverage_mean=$(echo "${coverage_scores[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i} END {print s/NF}')
#         coverage_std=$(echo "${coverage_scores[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i; sumsq+=$i*$i} END {print sqrt(sumsq/NF - (s/NF)^2)}')

#         # 打印结果
#         echo "For method: $method, parameter: $param, Mean QD score: $qd_mean, QD score STD: $qd_std"
#         echo "For method: $method, parameter: $param, Mean Coverage: $coverage_mean, Coverage STD: $coverage_std"

#         # 将 mean 和 std 添加到 CSV 文件中
#         echo "$method,$param,Mean,Mean,$qd_mean,$coverage_mean" >> $output_csv
#         echo "$method,$param,STD,STD,$qd_std,$coverage_std" >> $output_csv
#     done
# done

# echo "Experiment results have been saved to $output_csv"