# NER for gene, variant, disease, chemical

## Development Environment

- Docker image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

Environment setup
```bash
pip install -r requirements.txt
```

## How to Run
### 1. Prepare sentences
An example input file is available at `source.tsv`
```
Methods	O
for	O
direct	O
sequencing	O
and	O
mutational	O
analyses	O
were	O
described	O
.	O

Patients	O
with	O
biallelic	O
mutations	O
of	O
SLC26A4	O
tested	O
for	O
variants	O
within	O
the	O
KCNJ10	O
gene	O
...
```
### 2. Generate NER results

See `./demo.sh`

```bash
python main.py \
	--source_file ./source.tsv \
	--target_dir ./target_dir \
	--model_dir ./model_dir \
	--device cuda:0 \
	--batch_size 128 \
```