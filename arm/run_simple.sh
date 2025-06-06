#!/bin/bash


# python run_experiment1.py
# python main.py --seed 1111 --trial_id 53 --noisy_method noisy_labels_exact --parameter 0.3 --robust_loss label_smoothing
python main.py --seed 2222 --trial_id 53 --noisy_method noisy_labels_exact --parameter 0.3 --robust_loss robust_qdhf --device cuda --cuda_index 4
# python main.py --seed 5555 --trial_id 53 --noisy_method noisy_labels_exact --parameter 0.3 --robust_loss rDPO