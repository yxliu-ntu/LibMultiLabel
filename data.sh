#!/bin/bash

mkdir -p data/rcv1
cd data/rcv1

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/train_texts.txt.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/train_labels.txt.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/test_texts.txt.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/test_labels.txt.bz2

bzip2 -d *.bz2
cd ../..
