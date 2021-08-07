#!/bin/bash

set -x

python3 main.py --config example_config/curatedtrec/bert_tiny_2tower_dpr.yml --learning_rate 2e-5 --bsize_i 64 --bsize_j 64 --weight_decay 0.0
python3 main.py --config example_config/curatedtrec/bert_tiny_2tower_dpr_lrlr.yml --learning_rate 2e-5 --bsize_i 16 --bsize_j 16 --weight_decay 0.0 --omega 0.00390625
python3 main.py --config example_config/curatedtrec/bert_tiny_2tower_sogram.yml --learning_rate 2e-5 --bsize_i 16 --bsize_j 16 --weight_decay 0.0 --omega 6.103515625e-05 --imp_r -4
python3 main.py --config example_config/curatedtrec/bert_tiny_2tower_sogram_cosine.yml --learning_rate 2e-5 --bsize_i 16 --bsize_j 16 --weight_decay 0.0 --omega 0.00390625 --imp_r 0
python3 main.py --config example_config/curatedtrec/bert_tiny_2tower_sogram_scale.yml --learning_rate 2e-5 --bsize_i 16 --bsize_j 16 --weight_decay 0.0 --omega 0.00390625 --imp_r -1.0
