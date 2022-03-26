# Population Extraction

## Development Environment

- Python 3.6.9

| Package | Version |
|-|-|
| torch | 1.8.1+cu111 |
| transformers | 4.12.2 |
| nltk | 3.6.5 |
| datasets | 1.14.0 |
| sentencepiece | 0.1.96 |
| protobuf | 3.19.1 |
| rouge-score | 0.0.4 |

## How to Run

### 1. Download pre-trained models

- Git lfs is needed to clone large *pytorch_model.bin* model files and *.csv* dataset files.

### 2. Prepare sentences

An example input file is available at ./source.json

```json
[
  ...
  {
    "sentence": "Subjects were 14 normal healthy volunteers, 10 patients with spasmodic torticollis and 5 with essential tremor involving neck muscles.",
    "token_list": ["Subjects", "were", "14", "normal", "healthy", "volunteers", ",", "10", "patients", "with", "spasmodic", "torticollis", "and", "5", "with", "essential", "tremor", "involving", "neck", "muscles", "."],
    "pmid": "9851295",
    "sent_id": 2,
  },
  ...
]
```

### 3. Run extraction

See *./demo.sh*

It takes about 9 minutes to run the demo.

```bash
python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--model_dir ./model \
--number_measure_file ./number_measure.txt \
--species_file species.txt \
--indent 2
```

### 4. Results

Files:

- *./target_dir/model_input.csv*: a temporary file; can be deleted
- *./target_dir/model_output.txt*: a temporary file; can be deleted
- *./target_dir/all_results.json*: a temporary file; can be deleted
- *./target_dir/test_results.json*: a temporary file; can be deleted
- *./target_dir/target.json*: sentences with extracted populations

In *./target_dir/target.json*:

```json
[
  ...
  {
    "sentence": "Subjects were 14 normal healthy volunteers, 10 patients with spasmodic torticollis and 5 with essential tremor involving neck muscles.",
    "token_list": ["Subjects", "were", "14", "normal", "healthy", "volunteers", ",", "10", "patients", "with", "spasmodic", "torticollis", "and", "5", "with", "essential", "tremor", "involving", "neck", "muscles", "."],
    "pmid": "9851295",
    "sent_id": 2,
    "population": [
      "14 normal healthy volunteers",
      "10 patients",
      "5 with essential tremor involving neck muscles"
    ]
  },
  ...
]
```

### 6. Speed Profile

Environment:
- RTX 2080 ti (x1)

Dataset:
- 10,000 random abstracts from PubMed
- 95,441 sentences
- 2,028,796 words
- 2,339,053 tokens
- 13,914,110 characters

Time:
- 499 sec

