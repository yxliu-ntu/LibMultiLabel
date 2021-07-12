#!/bin/bash

root='log_compare'
mkdir -p $root

task(){
for i in 'Sogram-LRSQ' 'Minibatch-LRSQ' 'Naive-LRSQ' 'Naive-LRLR'
do
    for j in '' '--loaderFactory'
    do
        echo "python3 main.py --config example_config/sim/kim_cnn_2tower.yml --loss $i --silent $j > $root/${i}${j}.2tower.log"
    done
done
echo "python3 main.py --config example_config/sim/kim_cnn_2tower.yml --loss Ori-LRLR --silent > $root/Ori-LRLR.2tower.log"
echo "python3 main.py --config example_config/sim/kim_cnn.yml --loss Ori-LRLR --silent > $root/Ori-LRLR.log"
}

# Check command
task
wait

# Run
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
