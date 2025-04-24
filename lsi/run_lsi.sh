#!/bin/bash

# export CUDA_VISIBLE_DEVICES=5

python main.py --seed 2222 --noisy_method stochastic --parameter 0.3 --robust_loss None --device cpu --cuda_index 5
#python run_experiment.py