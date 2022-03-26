# pubmedKB Core

**End-to-end relation extraction for biomedical literature**

Authors:
- Peng-Hsuan Li (jacobvsdanniel [at] gmail.com)
- Ting-Fu Chen (ting830812 [at] gmail.com)

References:
- Original [BERN](https://github.com/dmis-lab/bern)
- Original [R-BERT](https://github.com/monologg/R-BERT)

## Introduction

- This is the core annotator behind:

*Peng-Hsuan Li, Ting-Fu Chen, Jheng-Ying Yu, Shang-Hung Shih, Chan-Hung Su, Yin-Hung Lin, Huai-Kuang Tsai, Hsueh-Fen Juan, Chien-Yu Chen and Jia-Hsin Huang, **pubmedKB: an interactive web server to explore biomedical entity relations from biomedical literature**, Nucleic Acids Research, under revision, 2022*

- It contains 7 modules:

| Module | Annotation | |
|-|-|-|
| NER | named entity recognition | including gene, variant, disease, chemical |
| NEN | named entity normalization | mapping aliases to unique IDs |
| POP | population extraction | *"1337 patients"* |
| OR | odds ratio extraction | *"variant: C.3630A>G, disease: PD, OR: 1.37, CI: 1.08-1.73, p-value: 0.008"* |
| SPACY | relational phrase extraction | *"AVED, causes, peripheral neuropathy"* |
| OPENIE | relational phrase extraction | *"pulmonary embolism, cause, pulmonary hypertention* |
| CRE | relational fact extraction | *"513insTT, in-patient, AVED"* |

- Visit our [website](https://www.pubmedkb.cc) for a knowledge base mined from more than 33M PubMed citations

![plot](https://github.com/jacobvsdanniel/pubmedkb_core/blob/master/image_dir/website.png)

## How to Run

### 0. Development Environment

- Python 3.6.9
- nltk 3.6.7

### 1. Download large files

Download [zip](https://drive.google.com/file/d/16LJViJvmSQLc6zbYK_MmwHrQyMi4tevB/view?usp=sharing)

Uncompress into:

```
demo_json_dir
├── ner_source.json -> move to model_dir/ner/source.json (if you want to run its demo)
├── nen_source.json -> move to model_dir/nen/source.json (if you want to run its demo)
├── pop_source.json -> move to model_dir/pop/source.json (if you want to run its demo)
├── or_source.json -> move to model_dir/or/source.json (if you want to run its demo)
├── spacy_source.json -> move to model_dir/spacy/source.json (if you want to run its demo)
├── openie_source.json -> move to model_dir/openie/source.json (if you want to run its demo)
└── cre_source.json -> move to model_dir/cre/source.json (if you want to run its demo)
model_bin_dir
├── ner_ner_model_dir -> move to model_dir/ner/ner/model_dir
├── pop_model -> move to model_dir/pop/model
├── or_model -> move to model_dir/or/model
└── cre_rbert_relation_extraction_model -> move to model_dir/cre/rbert_relation_extraction/model
```

### 2. Prepare core models

- Follow the instructions for each model in *model_dir*

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

- Be sure to download the nltk punkt model

```python
import nltk
nltk.download("punkt")
```

### 3. Create virtual environments

- Create virtual environments under *venv_dir*

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

### 4. Start nen server

Before running the core pipeline, the nen server must be started.

See the instructions in *model_dir/nen/nen_server*

### 4. Demo

See *demo.sh*.

```bash
python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--task ner,pop,or,spacy,nen,cre,openie \
--indent 2
```

### 5. Data format

The last six fields (mention_list, population, odds_ratio, spacy_ore, openie_ore, rbert_cre) can be absent in source files, and they will be filled in target files.

See the README in each *model_dir/module* for sample outputs.

```json
[
    ...
    {
        "pmid": "8602747",
        "sent_id": 6,
        "sentence": "The 2 patients with a severe form of AVED were homozygous with 485delT and 513insTT, respectively, while the patient with a mild form of the disease was compound heterozygous with 513insTT and 574G-->A.",
        "span_list": [[0, 3], [4, 5], [6, 14], [15, 19], [20, 21], [22, 28], [29, 33], [34, 36], [37, 41], [42, 46], [47, 57], [58, 62], [63, 70], [71, 74], [75, 83], [83, 84], [85, 97], [97, 98], [99, 104], [105, 108], [109, 116], [117, 121], [122, 123], [124, 128], [129, 133], [134, 136], [137, 140], [141, 148], [149, 152], [153, 161], [162, 174], [175, 179], [180, 188], [189, 192], [193, 197], [197, 199], [199, 200], [200, 201], [201, 202]],
        "token_list": ["The", "2", "patients", "with", "a", "severe", "form", "of", "AVED", "were", "homozygous", "with", "485delT", "and", "513insTT", ",", "respectively", ",", "while", "the", "patient", "with", "a", "mild", "form", "of", "the", "disease", "was", "compound", "heterozygous", "with", "513insTT", "and", "574G", "--", ">", "A", "."],
        "mention_list": [...],
        "population": [...],
        "odds_ratio": [...],
        "spacy_ore": [...],
        "openie_ore": [...],
        "rbert_cre": [...]
    },
    ...
]
```
