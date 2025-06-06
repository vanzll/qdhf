#!/bin/bash

# export CUDA_VISIBLE_DEVICES=5
#!/usr/bin/env bash
# File: run_tmux.sh
# Usage:  bash run_tmux.sh

SESSION="qd_hf_run"

# List the five noise methods you want to try
METHODS=(
  noisy_labels_exact
  stochastic
  add_equal_noise
  flip_by_distance
  flip_labels_asymmetric
)

# ---- create / configure tmux session ----
tmux new-session -d -s "$SESSION"
k=1
for i in "${!METHODS[@]}"; do
    
    METHOD="${METHODS[$i]}"
    WIN_NAME="$METHOD"

    if [[ $i -eq 0 ]]; then               # window 0 already exists
        tmux rename-window -t "$SESSION:0" "$WIN_NAME"
    else
        tmux new-window -t "$SESSION" -n "$WIN_NAME"
    fi

    # build the command string for this window
    CMD="source ~/.bashrc && conda activate QDHF && \
python main.py --noisy_method ${METHOD} --seed 2222 --parameter 0.3 \
               --robust_loss None --device cuda --cuda_index $(((k%3)+1))"

    # send the command and press <Enter>
    tmux send-keys -t "$SESSION:$WIN_NAME" "$CMD" C-m
    k=$((k+1))
    
done

# Finally, attach to the session so you can watch the runs
tmux attach -t "$SESSION"

#python run_experiment.py