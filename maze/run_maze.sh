#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py --seed 1234 --trial_id 10 --noisy_method stochastic --parameter 0.05

# python run_experiment.py