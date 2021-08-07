#!/bin/bash

# basic seting
config='example_config/curatedtrec/bert_tiny_2tower_dpr.yml'

task(){
# Set up train command
train_cmd="python3 main.py"
train_cmd="${train_cmd} --config ${config}"

# Print out all parameter pair
for br in 16 64 256 #1024
do
    for lr in 2e-5
    do
        for wd in 0.0
        do
            cmd="${train_cmd} --learning_rate ${lr}"
            #cmd="${cmd} --bratio ${br}"
            cmd="${cmd} --bsize_i ${br} --bsize_j ${br}"
            cmd="${cmd} --weight_decay ${wd}"
            echo "${cmd}"
        done
    done
done
}

# Check command
task
wait

# Run
#task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
