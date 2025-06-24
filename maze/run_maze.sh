#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python run_experiment.py --noise_type flip_labels_asymmetric --robust label_smoothing --device cpu