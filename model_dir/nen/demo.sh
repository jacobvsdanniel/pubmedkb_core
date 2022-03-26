#!/bin/bash

set -eux

python main.py \
--source_file source.json \
--target_file target.json \
--batch_size 10000 \
--processes 20 \
--port 8888 \
--indent 2 \
2>&1 | tee log.txt

