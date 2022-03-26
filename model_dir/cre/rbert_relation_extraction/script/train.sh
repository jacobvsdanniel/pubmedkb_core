#!/bin/bash

set -eux

base=$1
dataset=$2
log_steps=$3
train_steps=$4
lr_steps=$5

rm -f ./data/${dataset}/cached_*

export CUDA_VISIBLE_DEVICES=0

python main.py \
--do_train \
--do_eval \
--model_name_or_path ./model/${base} \
--tokenizer_model_name_or_path ./model/biobert \
--data_dir ./data/${dataset} \
--model_dir ./model/${base}-${dataset} \
--eval_dir ./eval/${dataset} \
--train_batch_size 16 \
--logging_steps ${log_steps} \
--save_steps ${train_steps} \
--max_steps ${train_steps} \
--lr_schedule_steps ${lr_steps} \
2>&1 | tee log/train_${base}-${dataset}_log${log_steps}_train${train_steps}_lr${lr_steps}.txt

