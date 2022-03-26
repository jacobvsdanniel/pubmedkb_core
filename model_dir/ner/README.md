# NER Tool

## Development Environment

- The same as *./ner*

*Warning: That repo requires cuda 10.1, which does NOT support RTX 3090.*

## How to Run

### 1. Prepare Model and NLTK Data

- Follow the instructions in *./ner*
- Download nltk data using python:
```python
import nltk
nltk.download("punkt")
```

### 2. Prepare Source Data

An example input file is available at ./source.json

```json
[
    ...
    {
        "pmid": "8602747",
        "sent_id": 6,
        "sentence": "The 2 patients with a severe form of AVED were homozygous with 485delT and 513insTT, respectively, while the patient with a mild form of the disease was compound heterozygous with 513insTT and 574G-->A.",
        "span_list": [[0, 3], [4, 5], [6, 14], [15, 19], [20, 21], [22, 28], [29, 33], [34, 36], [37, 41], [42, 46], [47, 57], [58, 62], [63, 70], [71, 74], [75, 83], [83, 84], [85, 97], [97, 98], [99, 104], [105, 108], [109, 116], [117, 121], [122, 123], [124, 128], [129, 133], [134, 136], [137, 140], [141, 148], [149, 152], [153, 161], [162, 174], [175, 179], [180, 188], [189, 192], [193, 197], [197, 199], [199, 200], [200, 201], [201, 202]],
        "token_list": ["The", "2", "patients", "with", "a", "severe", "form", "of", "AVED", "were", "homozygous", "with", "485delT", "and", "513insTT", ",", "respectively", ",", "while", "the", "patient", "with", "a", "mild", "form", "of", "the", "disease", "was", "compound", "heterozygous", "with", "513insTT", "and", "574G", "--", ">", "A", "."]
    }
    ...
]
```

### 3. Run

See *./demo.sh*

It takes about 7 minutes to run the demo.

```bash
python main.py \
--source_file source.json \
--target_dir target_dir \
--indent 2
```

### 4. Results

- *./target_dir/target.json*: sentences with NER annotations
- *./target_dir/input.txt*: a temporary file; can be deleted
- *./target_dir/target.tsv*: a temporary file; can be deleted

### 5. Result format

Possible NER types:
- gene
- mutation
- disease
- drug

Expected results in *./target_dir/target.json*:
```json
[
    ...
    {
        "pmid": "8602747",
        "sent_id": 6,
        "sentence": "The 2 patients with a severe form of AVED were homozygous with 485delT and 513insTT, respectively, while the patient with a mild form of the disease was compound heterozygous with 513insTT and 574G-->A.",
        "span_list": [[0, 3], [4, 5], [6, 14], [15, 19], [20, 21], [22, 28], [29, 33], [34, 36], [37, 41], [42, 46], [47, 57], [58, 62], [63, 70], [71, 74], [75, 83], [83, 84], [85, 97], [97, 98], [99, 104], [105, 108], [109, 116], [117, 121], [122, 123], [124, 128], [129, 133], [134, 136], [137, 140], [141, 148], [149, 152], [153, 161], [162, 174], [175, 179], [180, 188], [189, 192], [193, 197], [197, 199], [199, 200], [200, 201], [201, 202]],
        "token_list": ["The", "2", "patients", "with", "a", "severe", "form", "of", "AVED", "were", "homozygous", "with", "485delT", "and", "513insTT", ",", "respectively", ",", "while", "the", "patient", "with", "a", "mild", "form", "of", "the", "disease", "was", "compound", "heterozygous", "with", "513insTT", "and", "574G", "--", ">", "A", "."],
        "mention_list": [
            {"name": "AVED", "type": "disease", "pos": [8, 9], "real_pos": [37, 41]},
            {"name": "485delT", "type": "mutation", "pos": [12, 13], "real_pos": [63, 70]},
            {"name": "513insTT", "type": "mutation", "pos": [14, 15], "real_pos": [75, 83]},
            {"name": "513insTT", "type": "mutation", "pos": [32, 33], "real_pos": [180, 188]},
            {"name": "574G-->A", "type": "mutation", "pos": [34, 38], "real_pos": [193, 201]}
        ]
    },
    ...
]
```

### 6. Speed Profile

Environment:
- RTX 2080 Ti (x1)

Dataset:
- 10,000 random abstracts from PubMed
- 95,441 sentences
- 2,028,796 words
- 2,339,053 tokens
- 13,914,110 characters

Time:
- 383 sec
