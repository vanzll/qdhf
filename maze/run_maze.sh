#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# python main.py --seed 1111 --trial_id 50 --noisy_method stochastic --parameter 0.05

python run_experiment.py