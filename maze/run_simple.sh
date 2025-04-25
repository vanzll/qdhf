#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

python main.py --seed 2222 --trial_id 51 --noisy_method noisy_labels_exact --parameter 0.05 --robust_loss None --device cpu --cuda_index 0