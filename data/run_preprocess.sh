#!/usr/bin/env bash

if [ $1 == "EUR-Lex" ]; then
  TRAIN_TEXT="--text_path $1/train_texts.txt"
  TEST_TEXT="--text_path $1/test_texts.txt"
else
  TRAIN_TEXT="--text_path $1/train_raw_texts.txt"
  TEST_TEXT="--text_path $1/test_raw_texts.txt"
fi

python preprocess.py $TRAIN_TEXT --label_path $1/train_labels.txt --save_path $1/train.txt
python preprocess.py $TEST_TEXT  --label_path $1/test_labels.txt --save_path $1/test.txt
