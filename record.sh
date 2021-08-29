#!/bin/bash

set -x
#CUDA_VISIBLE_DEVICES=0 python3 main.py --record --max_steps 29962 \
#    --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr_triplet.yml \
#    --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0 --omega 1.0 --triplet_margin 8 --seed 1339 \
#    --validL_path data/nq.subsampled_no-extra-neg/trainL.csv --validR_path data/nq.subsampled_no-extra-neg/trainR.csv \
#    --result_dir ./runs/record --tfboard_log_dir tfboard_logs/record

#CUDA_VISIBLE_DEVICES=0 python3 main.py --record --max_steps 24566 \
#    --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr.yml \
#    --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0 --seed 1339 \
#    --validL_path data/nq.subsampled_no-extra-neg/trainL.csv --validR_path data/nq.subsampled_no-extra-neg/trainR.csv \
#    --result_dir ./runs/record --tfboard_log_dir tfboard_logs/record

#CUDA_VISIBLE_DEVICES=0 python3 main.py --record --max_steps 30530 \
#    --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr_lrsq.yml \
#    --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0 --omega 0.000244140625 --imp_r -30 --seed 1339 \
#    --validL_path data/nq.subsampled_no-extra-neg/trainL.csv --validR_path data/nq.subsampled_no-extra-neg/trainR.csv \
#    --result_dir ./runs/record --tfboard_log_dir tfboard_logs/record

CUDA_VISIBLE_DEVICES=0 python3 main.py --record --max_steps 44588 \
    --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr_mse.yml \
    --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0 --omega 0.00390625 --imp_r -1 --seed 1339 \
    --validL_path data/nq.subsampled_no-extra-neg/trainL.csv --validR_path data/nq.subsampled_no-extra-neg/trainR.csv \
    --result_dir ./runs/record --tfboard_log_dir tfboard_logs/record

#CUDA_VISIBLE_DEVICES=0 python3 main.py --record --max_steps 116156 \
#    --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr_l1hinge.yml \
#    --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0 --omega 0.000244140625 --imp_r -1 --seed 1339 \
#    --validL_path data/nq.subsampled_no-extra-neg/trainL.csv --validR_path data/nq.subsampled_no-extra-neg/trainR.csv \
#    --result_dir ./runs/record --tfboard_log_dir tfboard_logs/record

#CUDA_VISIBLE_DEVICES=0 python3 main.py --record --max_steps 92442 \
#    --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr_l2hinge.yml \
#    --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0 --omega 0.000244140625 --imp_r -1 --seed 1339 \
#    --validL_path data/nq.subsampled_no-extra-neg/trainL.csv --validR_path data/nq.subsampled_no-extra-neg/trainR.csv \
#    --result_dir ./runs/record --tfboard_log_dir tfboard_logs/record

#CUDA_VISIBLE_DEVICES=0 python3 main.py --record --max_steps 40186 \
#    --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr_lrlr.yml \
#    --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0 --omega 0.000244140625 --seed 1339 \
#    --validL_path data/nq.subsampled_no-extra-neg/trainL.csv --validR_path data/nq.subsampled_no-extra-neg/trainR.csv \
#    --result_dir ./runs/record --tfboard_log_dir tfboard_logs/record

