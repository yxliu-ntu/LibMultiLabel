#!/bin/bash

# basic seting
config=$1
gpu=$2

task(){
# Set up train command
train_cmd="CUDA_VISIBLE_DEVICES=$gpu python3 main.py"
train_cmd="${train_cmd} --config ${config}"
#train_cmd="${train_cmd} --isWithoutWeight"
wd=0.0

# Print out all parameter pair
for br in 64 #16 256 #1024
do
    for lr in 2e-5
    do
        #for seed in 1331 1333 1335 1337 1339
        for seed in 1339
        do
            cmd="${train_cmd} --learning_rate ${lr}"
            #cmd="${cmd} --bratio ${br}"
            cmd="${cmd} --bsize_i ${br} --bsize_j ${br}"
            cmd="${cmd} --weight_decay ${wd}"
            cmd="${cmd} --seed ${seed}"
            echo "${cmd}"
        done
    done
done
}

# Check command
task
wait

# Run
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
