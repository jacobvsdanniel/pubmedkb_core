# OpenIE ORE

## Development Environment

- Python 3.6.9
- nltk 3.6.2
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) 4.2.0

## How to Run

### 1. Prepare Stanford CoreNLP

From [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/), download the zipped release; unzip it to a path, say, *./jar_dir*

A copy of the unzipped jar directory is available at */volume/pubmedkb-covid/openie/stanford-corenlp-4.2.0*

### 2. Prepare sentences and NER annotations

An example input file is available at ./source.json

```json
[
    ...
    {
        "sentence": "It is demonstrated that recurrent pulmonary embolism may cause pulmonary hypertention.",
        "span_list": [[0, 2], [3, 5], [6, 18], [19, 23], [24, 33], [34, 43], [44, 52], [53, 56], [57, 62], [63, 72], [73, 85], [85, 86]],
        "token_list": ["It", "is", "demonstrated", "that", "recurrent", "pulmonary", "embolism", "may", "cause", "pulmonary", "hypertention", "."], "pmid": "7541326", "sent_id": 2,
        "mention_list": [
            {"name": "pulmonary embolism", "type": "DISEASE", "pos": [5, 7], "real_pos": [34, 52]},
            {"name": "pulmonary hypertention", "type": "DISEASE", "pos": [9, 11], "real_pos": [63, 85]}
        ]
    },
    ...
]
```

### 3. Run

See *./demo.sh*

It takes about 14 minutes to run the demo.

```bash
python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--jar_dir ./jar_dir/ \
--batch_size 2000 \
--batch_processes 4 \
--process_memory 6g \
--indent 2
```

### 4. Results

- *./target_dir/target.json*: sentences with triplets (relations)
- *./target_dir/all_source*: a temporary directory; can be deleted
- *./target_dir/all_target*: a temporary directory; can be deleted

### 5. Triplet format

- The head of a triplet always contains exactly one entity mention, so does the tail.
- The two entity mentions of head and tail are always different.
- If both the head and tail are equal to their respective entity mentions, perfect_match=True

A sample sentence with additional triplet annotations in *./target_dir/target.json*:
```json
[
    ...
    {
        "sentence": "It is demonstrated that recurrent pulmonary embolism may cause pulmonary hypertention.",
        "span_list": [[0, 2], [3, 5], [6, 18], [19, 23], [24, 33], [34, 43], [44, 52], [53, 56], [57, 62], [63, 72], [73, 85], [85, 86]],
        "token_list": ["It", "is", "demonstrated", "that", "recurrent", "pulmonary", "embolism", "may", "cause", "pulmonary", "hypertention", "."], "pmid": "7541326", "sent_id": 2,
        "mention_list": [
            {"name": "pulmonary embolism", "type": "DISEASE", "pos": [5, 7], "real_pos": [34, 52]},
            {"name": "pulmonary hypertention", "type": "DISEASE", "pos": [9, 11], "real_pos": [63, 85]}
        ],
        "triplet_list": [
            {"h_mention": 0, "t_mention": 1, "perfect_match": true, "triplet": ["pulmonary embolism", "cause", "pulmonary hypertention"]}
        ]
    },
    ...
]
```

### 6. Speed Profile

Environment:
- CPU only (1000%)

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

Time:
- 821 sec

## The pipeline

The pipeline tasks are, in order:

| Task | |
|-|-|
| read *data* | from *source.json* |
| preprocess | add masking to *data* |
| openie | generate raw triplets and add them to *data* |
| postprocess | match and correct triplets in *data* |
| write *data* | to *target_dir/target.json* |
