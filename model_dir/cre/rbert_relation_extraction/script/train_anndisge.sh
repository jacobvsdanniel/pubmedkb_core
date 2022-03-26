#!/bin/bash

set -eux

export CUDA_VISIBLE_DEVICES=0

python student.py \
--base biobert \
--gold anndisge \
--auto nonanndisge \
--retrain_gold_model \
--student_iterations 3 \
2>&1 | tee log/train_anndisge.txt

