#!/bin/bash

set -eux

python main.py \
--source_file source.json \
--target_dir target_dir \
--indent 2 \
2>&1 | tee log.txt

