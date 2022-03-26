#!/bin/bash

set -eux

python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--model_dir ./model \
--number_measure_file ./number_measure.txt \
--species_file species.txt \
--indent 2 \
2>&1 | tee log.txt

