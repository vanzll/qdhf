#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py --noisy_method noisy_labels_exact --parameter 0.5 --seed 1111
#python run_experiment.py