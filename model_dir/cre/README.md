# R-BERT CRE Tool

## Development Environment

This module uses the same environment as *./rbert_relation_extraction*

- Python 3.6.9
- torch 1.6.0+cu101
- transformers 3.3.1

*Warning: That repo requires cuda 10.1, which does NOT support RTX 3090.*

## How to Run

### 1. Prepare sentences and NER annotations

An example input file is available at ./source.json

```json
[
    ...
    {
        "sentence": "The 2 patients with a severe form of AVED were homozygous with 485delT and 513insTT, respectively, while the patient with a mild form of the disease was compound heterozygous with 513insTT and 574G-->A.",
        "token_list": ["The", "2", "patients", "with", "a", "severe", "form", "of", "AVED", "were", "homozygous", "with", "485delT", "and", "513insTT", ",", "respectively", ",", "while", "the", "patient", "with", "a", "mild", "form", "of", "the", "disease", "was", "compound", "heterozygous", "with", "513insTT", "and", "574G", "--", ">", "A", "."],
        "pmid": "8602747",
        "sent_id": 6,
        "span_list": [[0, 3], [4, 5], [6, 14], [15, 19], [20, 21], [22, 28], [29, 33], [34, 36], [37, 41], [42, 46], [47, 57], [58, 62], [63, 70], [71, 74], [75, 83], [83, 84], [85, 97], [97, 98], [99, 104], [105, 108], [109, 116], [117, 121], [122, 123], [124, 128], [129, 133], [134, 136], [137, 140], [141, 148], [149, 152], [153, 161], [162, 174], [175, 179], [180, 188], [189, 192], [193, 197], [197, 199], [199, 200], [200, 201], [201, 202]],
        "mention_list": [
            {"name": "AVED", "type": "disease", "pos": [8, 9], "real_pos": [37, 41], "id": []},
            {"name": "485delT", "type": "mutation", "pos": [12, 13], "real_pos": [63, 70], "id": [["BERN:2421304"]]},
            {"name": "513insTT", "type": "mutation", "pos": [14, 15], "real_pos": [75, 83], "id": [["BERN:2421404"]]},
            {"name": "513insTT", "type": "mutation", "pos": [32, 33], "real_pos": [180, 188], "id": [["BERN:2421404"]]},
            {"name": "574G-->A", "type": "mutation", "pos": [34, 38], "real_pos": [193, 201], "id": [["BERN:2421504"]]}
        ]
    },
    ...
]
```

### 2. Run

See *./demo.sh*

It takes about 17 seconds to run the demo.

```bash
python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--variant_tag mutation \
--disease_tag disease \
--indent 2
```

### 3. Results

- *./target_dir/target.json*: sentences with triplets (relations)
- *./target_dir/meta.csv*: a temporary file; can be deleted
- *./target_dir/input.txt*: a temporary file; can be deleted
- *./target_dir/output.txt*: a temporary file; can be deleted
- *./target_dir/output.txt.npy*: a temporary file; can be deleted

### 4. Triplet format

- By default, the head of a triplet is a variant (mutation) and the tail is a disease, change *--variant_tag* and *--disease_tag* to modify model behavior
- A score of 0%-100% is also reported

Sample sentences with additional triplet annotations in *./target_dir/target.json*:
```json
[
    ...
    {
        "sentence": "The 2 patients with a severe form of AVED were homozygous with 485delT and 513insTT, respectively, while the patient with a mild form of the disease was compound heterozygous with 513insTT and 574G-->A.",
        "token_list": ["The", "2", "patients", "with", "a", "severe", "form", "of", "AVED", "were", "homozygous", "with", "485delT", "and", "513insTT", ",", "respectively", ",", "while", "the", "patient", "with", "a", "mild", "form", "of", "the", "disease", "was", "compound", "heterozygous", "with", "513insTT", "and", "574G", "--", ">", "A", "."],
        "pmid": "8602747",
        "sent_id": 6,
        "span_list": [[0, 3], [4, 5], [6, 14], [15, 19], [20, 21], [22, 28], [29, 33], [34, 36], [37, 41], [42, 46], [47, 57], [58, 62], [63, 70], [71, 74], [75, 83], [83, 84], [85, 97], [97, 98], [99, 104], [105, 108], [109, 116], [117, 121], [122, 123], [124, 128], [129, 133], [134, 136], [137, 140], [141, 148], [149, 152], [153, 161], [162, 174], [175, 179], [180, 188], [189, 192], [193, 197], [197, 199], [199, 200], [200, 201], [201, 202]],
        "mention_list": [
            {"name": "AVED", "type": "disease", "pos": [8, 9], "real_pos": [37, 41], "id": []},
            {"name": "485delT", "type": "mutation", "pos": [12, 13], "real_pos": [63, 70], "id": [["BERN:2421304"]]},
            {"name": "513insTT", "type": "mutation", "pos": [14, 15], "real_pos": [75, 83], "id": [["BERN:2421404"]]},
            {"name": "513insTT", "type": "mutation", "pos": [32, 33], "real_pos": [180, 188], "id": [["BERN:2421404"]]},
            {"name": "574G-->A", "type": "mutation", "pos": [34, 38], "real_pos": [193, 201], "id": [["BERN:2421504"]]}
        ],
        "triplet_list": [
            {
                "h_mention": [1],
                "t_mention": [0],
                "score": [["other", "15.0%"], ["cause-positive", "0.0%"], ["appositive", "2.6%"], ["in-patient", "82.4%"]],
                "triplet": ["485delT", "in-patient", "AVED"]
            },
            {
                "h_mention": [2, 3],
                "t_mention": [0],
                "score": [["other", "23.2%"], ["cause-positive", "0.0%"], ["appositive", "3.5%"], ["in-patient", "73.3%"]],
                "triplet": ["513insTT", "in-patient", "AVED"]
            },
            {
                "h_mention": [4],
                "t_mention": [0],
                "score": [["other", "27.2%"], ["cause-positive", "0.0%"], ["appositive", "7.3%"], ["in-patient", "65.5%"]],
                "triplet": ["574G-->A", "in-patient", "AVED"]
            }
        ]
    },
    ...
]
```

### 5. Speed Profile

Environment:
- RTX 2080 ti (x1)

Dataset:
- 10,000 random abstracts from PubMed
- 95,441 sentences
- 2,028,796 words
- 2,339,053 tokens
- 13,914,110 characters
- 7,519 genes
- 562 mutations
- 32,834 diseases
- 33,165 drugs
- 234 variant-disease-sentence samples to classify

Time:
- 18 sec
