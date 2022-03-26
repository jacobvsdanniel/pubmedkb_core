#!/bin/bash

set -eux

python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--model_dir ./model \
--variant_tag mutation \
--disease_tag disease \
--complete_triplet_only \
--batch_size 512 \
--indent 2 \
2>&1 | tee log.txt

