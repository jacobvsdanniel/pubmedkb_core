#!/bin/bash

set -eux

python main.py \
	--source_file ./source.tsv \
	--target_dir ./target_dir \
	--model_dir ./model_dir \
	--device cuda:0 \
	--batch_size 128 \
2>&1 | tee log.txt

