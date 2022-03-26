#!/bin/bash

set -eux

python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--variant_tag mutation \
--disease_tag disease \
--indent 2 \
2>&1 | tee log.txt

