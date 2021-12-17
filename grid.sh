#!/bin/bash

# basic seting
config=$1
loss=$2
gpu=$3

solver="adagrad"
lr=1e-2
epochs=300
br=1e-2
k=128
m=0
wd=0.0
r=-1.0

task(){
# Set up train command
train_cmd="python3 main.py"
train_cmd="${train_cmd} --optimizer $solver"
train_cmd="${train_cmd} --config ${config}"
train_cmd="${train_cmd} --learning_rate ${lr}"
train_cmd="${train_cmd} --epochs ${epochs}"
#train_cmd="${train_cmd} --bsize_i ${bs} --bsize_j ${bs}"
train_cmd="${train_cmd} --k ${k}"
train_cmd="${train_cmd} --momentum ${m}"
train_cmd="${train_cmd} --weight_decay ${wd}"
train_cmd="${train_cmd} --loss ${loss}"
train_cmd="${train_cmd} --isl2norm"
train_cmd="${train_cmd} --cpu"
train_cmd="${train_cmd} --close_early_stop"
train_cmd="${train_cmd} --check_func_val"
train_cmd="${train_cmd} --result_dir ./runs/"

# Print out all parameter pair
for seed in 1331 #1333 1335 1337 1339
do
    if [[ "$loss" =~ ^(Naive-LRLR)$ ]]; then
        for omega in 1 0.25 0.0625 0.015625 0.00390625 0.0009765625 #0.000244140625
        do
            for l in 0.25 1 4 16
            do
            	cmd="${train_cmd} --seed ${seed}"
            	cmd="${cmd} --omega ${omega}"
            	cmd="${cmd} --l2_lambda ${l}"
            	cmd="${cmd} --bratio ${br}"
            	echo "${cmd}"
			done
        done
    elif [[ "$loss" =~ ^(Naive-LRSQ|Sogram-LRSQ|Minibatch-LRSQ)$ ]]; then
        for omega in 0.0625 #0.0625 0.015625 0.00390625 0.0009765625 0.000244140625 6.103515625e-05 1.52587890625e-05
        do
            for l in 1 #0.25 1 4 16
            do
                cmd="${train_cmd} --seed ${seed}"
            	cmd="${cmd} --bratio ${br}"
                cmd="${cmd} --omega ${omega}"
                cmd="${cmd} --imp_r ${r}"
                cmd="${cmd} --l2_lambda ${l}"
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
#task | xargs -0 -d '\n' -P 2 -I {} sh -c {}
