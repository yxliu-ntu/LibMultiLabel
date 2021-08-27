#!/bin/bash

max_steps=29962

set -x
CUDA_VISIBLE_DEVICES=0 python3 main.py --record --max_steps ${max_steps} \
    --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr_triplet.yml \
    --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0 --omega 1.0 --triplet_margin 8 --seed 1339 \
    --validL_path data/nq.subsampled_no-extra-neg/trainL.csv --validR_path data/nq.subsampled_no-extra-neg/trainR.csv \
    --result_dir ./runs/record --tfboard_log_dir tfboard_logs/record
