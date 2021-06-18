#!/bin/bash

mkdir -p data/rcv1
cd data/rcv1

wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2
wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2

bzip2 -d *.bz2
cd ../..
