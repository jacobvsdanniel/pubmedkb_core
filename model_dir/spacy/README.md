# Spacy ORE

## Development Environment

- Python 3.6.9
- nltk 3.6.7
- tqdm 4.62.3
- spacy 3.2.1
- cupy-cuda101 9.6.0
- en-core-web-sm 3.2.0

## How to Run

### 1. Prepare spaCy

To install the lastest spacy and en-core-web-sm:
```bash
pip install -U spacy[cuda101]
python -m spacy download en_core_web_sm
```
*[cuda101]* should be modified to the CUDA version in the environment, or ignored if CUDA is not available.

### 2. Prepare sentences and NER annotations

An example input file is available at ./source.json

```json
[
    ...
    {
        "pmid": "8602747",
        "sent_id": 1,
        "sentence": "Familial vitamin E deficiency (AVED) causes ataxia and peripheral neuropathy that resembles Friedreich's ataxia.",
        "span_list": [[0, 8], [9, 16], [17, 18], [19, 29], [30, 31], [31, 35], [35, 36], [37, 43], [44, 50], [51, 54], [55, 65], [66, 76], [77, 81], [82, 91], [92, 102], [102, 104], [105, 111], [111, 112]],
        "token_list": ["Familial", "vitamin", "E", "deficiency", "(", "AVED", ")", "causes", "ataxia", "and", "peripheral", "neuropathy", "that", "resembles", "Friedreich", "'s", "ataxia", "."],
        "mention_list": [
            {"name": "AVED", "type": "DISEASE", "pos": [5, 6], "real_pos": [31, 35]},
            {"name": "ataxia", "type": "DISEASE", "pos": [8, 9], "real_pos": [44, 50]},
            {"name": "peripheral neuropathy", "type": "DISEASE", "pos": [10, 12], "real_pos": [55, 76]},
            {"name": "Friedreich's ataxia", "type": "DISEASE", "pos": [14, 17], "real_pos": [92, 111]}
        ]
    },
    ...
]
```

### 3. Run

See *./demo.sh*

It takes about 3 minutes to run the demo.

```bash
python main.py \
--source_file source.json \
--target_file target.json \
--batch_size 50000 \
--indent 2
```

### 4. Results

- *target.json*: sentences with triplets (relations)

### 5. Triplet format

- The head of a triplet always contains exactly one entity mention, so does the tail.
- The two entity mentions of head and tail are always different.
- If both the head and tail are equal to their respective entity mentions, perfect_match=True

A sample sentence with additional triplet annotations in *./target_dir/target.json*:
```json
[
    ...
    {
        "pmid": "8602747",
        "sent_id": 1,
        "sentence": "Familial vitamin E deficiency (AVED) causes ataxia and peripheral neuropathy that resembles Friedreich's ataxia.",
        "span_list": [[0, 8], [9, 16], [17, 18], [19, 29], [30, 31], [31, 35], [35, 36], [37, 43], [44, 50], [51, 54], [55, 65], [66, 76], [77, 81], [82, 91], [92, 102], [102, 104], [105, 111], [111, 112]],
        "token_list": ["Familial", "vitamin", "E", "deficiency", "(", "AVED", ")", "causes", "ataxia", "and", "peripheral", "neuropathy", "that", "resembles", "Friedreich", "'s", "ataxia", "."],
        "mention_list": [
            {"name": "AVED", "type": "DISEASE", "pos": [5, 6], "real_pos": [31, 35]},
            {"name": "ataxia", "type": "DISEASE", "pos": [8, 9], "real_pos": [44, 50]},
            {"name": "peripheral neuropathy", "type": "DISEASE", "pos": [10, 12], "real_pos": [55, 76]},
            {"name": "Friedreich's ataxia", "type": "DISEASE", "pos": [14, 17], "real_pos": [92, 111]}
        ],
        "triplet_list": [
            {"h_mention": 0, "t_mention": 1, "triplet": ["AVED", "causes", "ataxia"], "perfect_match": true},
            {"h_mention": 0, "t_mention": 2, "triplet": ["AVED", "causes", "peripheral neuropathy"], "perfect_match": true}
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
- 7,519 genes
- 562 mutations
- 32,834 diseases
- 33,165 drugs

Time:
- 40 sec
