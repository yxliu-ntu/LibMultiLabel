#!/bin/bash

# basic seting
config=$1
gpu=$2

task(){
# Set up train command
train_cmd="CUDA_VISIBLE_DEVICES=$gpu python3 main.py"
train_cmd="${train_cmd} --config ${config}"
#train_cmd="${train_cmd} --fix_q_encoder --fix_ctx_encoder"
train_cmd="${train_cmd} --result_dir ./runs/embedding_std"
train_cmd="${train_cmd} --tfboard_log_dir ./tfboard_logs/embedding_std"
#train_cmd="${train_cmd} --isWithoutWeight"
#train_cmd="${train_cmd} --projection_dim 128"
train_cmd="${train_cmd} --imp_r_shift 6.375143237143622"
train_cmd="${train_cmd} --embedding_scaler 12.708224895712226"
wd=0.0

# Print out all parameter pair
for br in 64 #16 64 256 1024
do
    for lr in 2e-5
    do
        #for seed in 1331 1333 1335 1337 1339
        for seed in 1339
        do
            for omega in 1.0 0.0625 0.00390625 0.000244140625
            #for omega in 6.103515625e-05 1.52587890625e-05 3.814697265625e-06 9.5367431640625e-07 2.384185791015625e-07
            do
                for r in 0 -1 -2 -4 -8 #-5 -10 -15 -20 -25 -30 -35 -40 -45
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
