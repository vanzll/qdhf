#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python main.py --seed 2222 --trial_id 53 --noisy_method stochastic --parameter 0.3 --robust_loss rDPO --device cpu --cuda_index 7