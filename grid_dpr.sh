#!/bin/bash

# basic seting
config=$1
loss=$2
gpu=$3

lr=2e-5
bs=64
wd=0.0

task(){
# Set up train command
train_cmd="CUDA_VISIBLE_DEVICES=$gpu python3 main.py"
train_cmd="${train_cmd} --config ${config}"
train_cmd="${train_cmd} --learning_rate ${lr}"
train_cmd="${train_cmd} --bsize_i ${bs} --bsize_j ${bs}"
train_cmd="${train_cmd} --weight_decay ${wd}"
train_cmd="${train_cmd} --loss ${loss}"
#train_cmd="${train_cmd} --fix_q_encoder --fix_ctx_encoder"
train_cmd="${train_cmd} --result_dir ./runs/rerun"
train_cmd="${train_cmd} --tfboard_log_dir ./tfboard_logs/rerun"
#train_cmd="${train_cmd} --isWithoutWeight"
#train_cmd="${train_cmd} --projection_dim 128"

# Print out all parameter pair
for seed in 1331 1333 1335 1337 1339
do
    cmd="${train_cmd} --seed ${seed}"
    if [[ "$loss" =~ ^(DPR-L1HL1H|DPR-L2HL2H|DPR-LRLR)$ ]]; then
        for omega in 0.0625 0.015625 0.00390625 0.0009765625 0.000244140625 6.103515625e-05 1.52587890625e-05
        do
            cmd="${train_cmd} --seed ${seed}"
            cmd="${cmd} --omega ${omega}"
            echo "${cmd}"
        done
    elif [[ "$loss" =~ ^(DPR-L1HSQ|DPR-L2HSQ|DPR-LRSQ)$ ]]; then
        for omega in 0.0625 0.015625 0.00390625 0.0009765625 0.000244140625 6.103515625e-05 1.52587890625e-05
        do
            for r in -5 -10 -15 -20 -25 -30 -35
            do
                cmd="${train_cmd} --seed ${seed}"
                cmd="${cmd} --omega ${omega}"
                cmd="${cmd} --imp_r ${r}"
                echo "${cmd}"
            done
        done
    elif [[ "$loss" =~ ^(DPR-RankingMSE|DPR-Triplet)$ ]]; then
        for m in 1 2 4 8 16 32 64
        do
            cmd="${train_cmd} --seed ${seed}"
            cmd="${cmd} --margin ${m}"
            echo "${cmd}"
        done
    elif [[ "$loss" =~ ^(DPR|DPR-L2Dist)$ ]]; then
        echo "${cmd}"
    else
        echo "Don't support ${loss}!"
        exit 0
    fi
done
}

# Check command
task
wait

# Run
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
