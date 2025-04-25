#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python run_experiment.py --noise_type noisy_labels_exact --robust reweight --device cpu