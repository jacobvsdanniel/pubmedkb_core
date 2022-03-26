# pubmedKB Core

Authors:
- Peng-Hsuan Li (jacobvsdanniel [at] gmail.com)
- Ting-Fu Chen (ting830812 [at] gmail.com)

References:
- original [BERN](https://github.com/dmis-lab/bern)
- original [R-BERT](https://github.com/monologg/R-BERT)

## Development Environment

- Python 3.6.9
- nltk 3.6.7

## How to Run

### 1. Prepare core models

- Follow the instructions for each model under *model_dir*.

```
model_dir/
├── ner
├── nen
├── pop
├── or
├── spacy
├── openie
└── cre
```

- Please put virtual environments for each model under *venv_dir*.

```
venv_dir/
├── ner
├── nen
├── pop
├── or
├── spacy
├── openie
└── cre
```

- Please make sure the nltk punkt package is downloaded

```python
import nltk
nltk.download("punkt")
```

### 2. Start nen server

Before running the core pipeline, the nen server must be started.

See the instructions in *model_dir/nen/nen_server*.

### 3. demo

See *demo.sh*.

```bash
python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--task ner,pop,or,spacy,nen,cre,openie \
--indent 2
```

