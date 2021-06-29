#!/bin/bash

root='log_compare'
mkdir -p $root

task(){
for i in 'Sogram-LRSQ' 'Minibatch-LRSQ' 'Naive-LRSQ' 'Naive-LRLR' 'Ori-LRLR'
do
    echo "python3 main.py --config example_config/sim/kim_cnn_2tower.yml --loss $i --silent > $root/$i.2tower.log"
done
echo "python3 main.py --config example_config/sim/kim_cnn.yml --loss $i --silent > $root/Ori-LRLR.log"
}

# Check command
task
wait

# Run
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
