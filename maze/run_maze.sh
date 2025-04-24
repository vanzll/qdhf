#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python run_experiment.py --noise_type stochastic --robust None --device cpu
# noisy_labels_exact stochastic flip_labels_asymmetric
# source .venv/bin/activate

# ./run_maze.sh