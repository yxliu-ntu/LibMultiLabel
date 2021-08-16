#!/bin/bash

# basic seting
config='example_config/curatedtrec/bert_tiny_2tower_dpr_lrsq.yml'

task(){
# Set up train command
train_cmd="python3 main.py"
train_cmd="${train_cmd} --config ${config}"
wd=0.0

# Print out all parameter pair
for br in 16 64 256 #1024
do
    for lr in 2e-5
    do
        for seed in 1331 1333 1335 1337 1339
        do
            for omega in 1.0 0.0625 0.00390625 0.000244140625
            do
                for r in -28 -30.0 -32.0
                do
                    cmd="${train_cmd} --learning_rate ${lr}"
                    #cmd="${cmd} --bratio ${br}"
                    cmd="${cmd} --bsize_i ${br} --bsize_j ${br}"
                    cmd="${cmd} --weight_decay ${wd}"
                    cmd="${cmd} --omega ${omega}"
                    cmd="${cmd} --imp_r ${r}"
                    cmd="${cmd} --seed ${seed}"
                    echo "${cmd}"
                done
            done
        done
    done
done
}

# Check command
task
wait

# Run
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
