#!/bin/bash

# basic seting
config=$1
loss=$2
gpu=$3

lr=2e-5
bs=4096
wd=0.0

task(){
# Set up train command
train_cmd="CUDA_VISIBLE_DEVICES=$gpu python3 main.py"
train_cmd="${train_cmd} --config ${config}"
train_cmd="${train_cmd} --learning_rate ${lr}"
train_cmd="${train_cmd} --bsize_i ${bs} --bsize_j ${bs}"
train_cmd="${train_cmd} --weight_decay ${wd}"
train_cmd="${train_cmd} --loss ${loss}"
train_cmd="${train_cmd} --result_dir ./runs/"
#train_cmd="${train_cmd} --tfboard_log_dir ./tfboard_logs/rerun"

# Print out all parameter pair
for seed in 1331 #1333 1335 1337 1339
do
    cmd="${train_cmd} --seed ${seed}"
    if [[ "$loss" =~ ^(Navie-LogSoftmax)$ ]]; then
        cmd="${train_cmd} --seed ${seed}"
        echo "${cmd}"
    elif [[ "$loss" =~ ^(Naive-LRLR)$ ]]; then
        for omega in 1 #0.0625 0.015625 0.00390625 0.0009765625 0.000244140625 6.103515625e-05 1.52587890625e-05
        do
            cmd="${train_cmd} --seed ${seed}"
            cmd="${cmd} --omega ${omega}"
            echo "${cmd}"
        done
    elif [[ "$loss" =~ ^(Naive-LRSQ|Sogram-LRSQ|Minibatch-LRSQ)$ ]]; then
        for omega in 1 #0.0625 0.015625 0.00390625 0.0009765625 0.000244140625 6.103515625e-05 1.52587890625e-05
        do
            for r in 0 #-1 -2 -4
            do
                cmd="${train_cmd} --seed ${seed}"
                cmd="${cmd} --omega ${omega}"
                cmd="${cmd} --imp_r ${r}"
                echo "${cmd}"
            done
        done
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
