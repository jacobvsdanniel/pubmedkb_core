#!/bin/bash

set -eux

src=$1
tgt=$2

rm -f ./data/${tgt}/cached_*
rm -f ./data/${tgt}/test_sentence.txt
rm -f ./eval/${tgt}/proposed_answers.txt
rm -f ./eval/${tgt}/result.txt

export CUDA_VISIBLE_DEVICES=0

python predeval.py \
--source ${src} \
--target ${tgt} \
2>&1 | tee log/predeval_model-${src}_dataset-${tgt}.txt

