#!/bin/bash


# python run_experiment1.py
# python main.py --seed 1111 --trial_id 53 --noisy_method noisy_labels_exact --parameter 0.3 --robust_loss label_smoothing
# python main.py --seed 1111 --trial_id 52 --noisy_method noisy_labels_exact --parameter 0.2 --robust_loss robust_qdhf --device cuda --cuda_index 5
python main.py --seed 7777 --trial_id 53 --noisy_method noisy_labels_exact --parameter 0.3 --robust_loss log_exp --device cuda --cuda_index 5
# python main.py --seed 2222 --trial_id 53 --noisy_method None --parameter 0.3 --robust_loss None --device cuda --cuda_index 5
# python main.py --seed 5555 --trial_id 53 --noisy_method noisy_labels_exact --parameter 0.3 --robust_loss rDPO