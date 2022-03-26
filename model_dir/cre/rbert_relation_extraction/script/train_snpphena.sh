#!/bin/bash

set -eux

export CUDA_VISIBLE_DEVICES=0

python student.py \
--base biobert \
--gold snpphena \
--auto disgevar \
--inherit_teacher \
--retrain_gold_model \
--student_iterations 2 \
2>&1 | tee log/train_snpphena.txt

