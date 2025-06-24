#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python run_experiment2.py --noise_type flip_labels_asymmetric --robust log_exp --device cuda
# noisy_labels_exact stochastic flip_labels_asymmetric
# source .venv/bin/activate