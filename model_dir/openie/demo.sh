#!/bin/bash

set -eux

python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--jar_dir ./jar_dir/ \
--batch_size 2000 \
--batch_processes 4 \
--process_memory 6g \
--indent 2 \
2>&1 | tee log.txt

