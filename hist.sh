#!/bin/bash

for i in 'lrlr' 'lrsq' 'mael1hinge' 'l1hingemae' 'l1hinge' 'l2hinge' 'mse' 'mae' 'l2hingesq' 'sql2hinge'
do
    python3 main.py --config example_config/nq.subsampled_no-extra-neg/bert_tiny_2tower_dpr_${i}.yml --eval --checkpoint_path runs/pos_neg_diff/nq.subsampled_no-extra-neg_bert_tiny_2tower_dpr_${i}_*/best_model.ckpt
done
