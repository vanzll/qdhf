#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python run_experiment2.py --noise_type noisy_labels_exact --robust cDPO --device cuda
# noisy_labels_exact stochastic flip_labels_asymmetric
# source .venv/bin/activate