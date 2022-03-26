#!/bin/bash

set -eux

python main.py \
--source_file source.json \
--target_file target.json \
--batch_size 50000 \
--indent 2 \
2>&1 | tee log.txt

