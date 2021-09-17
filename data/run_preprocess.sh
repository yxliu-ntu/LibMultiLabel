#!/usr/bin/env bash

if [ $1 == "EUR-Lex" ]; then
  TRAIN_TEXT="$1/train_texts.txt"
  TEST_TEXT="$1/test_texts.txt"
else
  TRAIN_TEXT="$1/train_raw_texts.txt"
  TEST_TEXT="$1/test_raw_texts.txt"
fi

## To txt
#python preprocess.py --text_path $TRAIN_TEXT --label_path $1/train_labels.txt --save_path $1/train.txt
#python preprocess.py --text_path $TEST_TEXT  --label_path $1/test_labels.txt --save_path $1/test.txt

## To csv
cat $1/train_labels.txt $1/test_labels.txt > $1/train_test_labels.txt
cat $TRAIN_TEXT $TEST_TEXT > $1/train_test_texts.txt
train_num=`wc -l $TRAIN_TEXT | cut -d' ' -f1`
test_num=`wc -l $TEST_TEXT | cut -d' ' -f1`

python preprocess.py --mode csv --text_path $1/train_test_texts.txt --label_path $1/train_test_labels.txt --save_path $1/train_test.csv --save_label_path $1/label.csv
head -n $train_num $1/train_test.csv > $1/train.csv
tail -n $test_num $1/train_test.csv > $1/test.csv
