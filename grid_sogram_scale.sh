#!/bin/bash

# basic seting
config='example_config/curatedtrec/bert_tiny_2tower_sogram_scale.yml'

task(){
# Set up train command
train_cmd="python3 main.py"
#train_cmd="${train_cmd} --eval_sqrt_mode"
train_cmd="${train_cmd} --config ${config}"

# Print out all parameter pair
for br in 16 64 256 #1024
do
    for lr in 2e-5
    do
        for wd in 0.0
        do
            for omega in 0.00390625 0.0009765625 0.000244140625 0.00006103515625 1.0 0.25 0.0625 0.015625
            do
                for r in -1.0 -4.0 -16.0 -64.0
                do
                    cmd="${train_cmd} --learning_rate ${lr}"
                    #cmd="${cmd} --bratio ${br}"
                    cmd="${cmd} --bsize_i ${br} --bsize_j ${br}"
                    cmd="${cmd} --weight_decay ${wd}"
                    cmd="${cmd} --omega ${omega}"
                    cmd="${cmd} --imp_r ${r}"
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
#task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
