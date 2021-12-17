#!/bin/bash

# basic seting
config=$1
loss=$2
gpu=$3

solver="adagrad"
#solver="sgd"
lr=1e-02
epochs=300
#total_steps=100000000
br=1
k=1
#omega=0.015625
#l=4
m=0
wd=0.0

task(){
# Set up train command
train_cmd="python3 main.py"
train_cmd="${train_cmd} --optimizer $solver"
train_cmd="${train_cmd} --config ${config}"
train_cmd="${train_cmd} --epochs ${epochs}"
#train_cmd="${train_cmd} --total_steps ${total_steps}"
#train_cmd="${train_cmd} --bsize_i ${bs}"
train_cmd="${train_cmd} --k ${k}"
train_cmd="${train_cmd} --momentum ${m}"
train_cmd="${train_cmd} --weight_decay ${wd}"
train_cmd="${train_cmd} --loss ${loss}"
train_cmd="${train_cmd} --isl2norm"
train_cmd="${train_cmd} --cpu"
train_cmd="${train_cmd} --close_early_stop"
#train_cmd="${train_cmd} --check_func_val"
train_cmd="${train_cmd} --result_dir ./runs/"

# Print out all parameter pair
for seed in 1331 #1333 1335 1337 1339
do
    cmd="${train_cmd} --seed ${seed}"
    if [[ "$loss" =~ ^(Linear-LR)$ ]]; then
        for omega in 1 #0.25 0.0625 0.015625 0.00390625 0.0009765625 #0.000244140625
        do
            for l in 16 #4 1 0.25
            do
                cmd="${train_cmd} --seed ${seed}"
                cmd="${cmd} --l2_lambda ${l}"
                #cmd="${cmd} --bsize_i ${bs}"
                cmd="${cmd} --bratio ${br}"
                cmd="${cmd} --learning_rate ${lr}"
                cmd="${cmd} --omega ${omega}"
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
task | xargs -0 -d '\n' -P 2 -I {} sh -c {}
