# Odds Ratio Extraction

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

### 2. Prepare sentences and NER annotations

An example input file is available at ./source.json

```json
[
    ...
    {
        "pmid": "25279164",
        "sent_id": 5,
        "sentence": "The results indicated that the GIGYF2 C.3630A>G polymorphism increased the risk of PD by 37% [P=0.008; odds ratio (OR), 1.37; 95% confidence interval (CI), 1.08-1.73] and that the GIGYF2 C.167G>A polymorphism was significantly associated with PD (P=0.003; OR, 3.67; 95% CI, 1.56-8.68).",
        "span_list": [[0, 3], [4, 11], [12, 21], [22, 26], [27, 30], [31, 37], [38, 45], [45, 46], [46, 47], [48, 60], [61, 70], [71, 74], [75, 79], [80, 82], [83, 85], [86, 88], [89, 91], [91, 92], [93, 94], [94, 101], [101, 102], [103, 107], [108, 113], [114, 115], [115, 117], [117, 118], [118, 119], [120, 124], [124, 125], [126, 128], [128, 129], [130, 140], [141, 149], [150, 151], [151, 153], [153, 154], [154, 155], [156, 165], [165, 166], [167, 170], [171, 175], [176, 179], [180, 186], [187, 193], [193, 194], [194, 195], [196, 208], [209, 212], [213, 226], [227, 237], [238, 242], [243, 245], [246, 247], [247, 254], [254, 255], [256, 258], [258, 259], [260, 264], [264, 265], [266, 268], [268, 269], [270, 272], [272, 273], [274, 283], [283, 284], [284, 285]],
        "token_list": ["The", "results", "indicated", "that", "the", "GIGYF2", "C.3630A", ">", "G", "polymorphism", "increased", "the", "risk", "of", "PD", "by", "37", "%", "[", "P=0.008", ";", "odds", "ratio", "(", "OR", ")", ",", "1.37", ";", "95", "%", "confidence", "interval", "(", "CI", ")", ",", "1.08-1.73", "]", "and", "that", "the", "GIGYF2", "C.167G", ">", "A", "polymorphism", "was", "significantly", "associated", "with", "PD", "(", "P=0.003", ";", "OR", ",", "3.67", ";", "95", "%", "CI", ",", "1.56-8.68", ")", "."],
        "mention_list": [
            {"name": "GIGYF2", "type": "gene", "pos": [5, 6], "real_pos": [31, 37]},
            {"name": "C.3630A>G", "type": "mutation", "pos": [6, 9], "real_pos": [38, 47]},
            {"name": "PD", "type": "disease", "pos": [14, 15], "real_pos": [83, 85]},
            {"name": "GIGYF2", "type": "gene", "pos": [42, 43], "real_pos": [180, 186]},
            {"name": "C.167G>A", "type": "mutation", "pos": [43, 46], "real_pos": [187, 195]},
            {"name": "PD", "type": "disease", "pos": [51, 52], "real_pos": [243, 245]}
        ]
    },
    ...
]
```

### 3. Run extraction

See *./demo.sh*

It takes about 4 minutes to run the demo.

```bash
python main.py \
--source_file ./source.json \
--target_dir ./target_dir \
--model_dir ./model \
--variant_tag mutation \
--disease_tag disease \
--complete_triplet_only \
--batch_size 512 \
--indent 2
```

### 4. Results

Files:

- *./target_dir/all_source*: a temporary directory; can be deleted
- *./target_dir/all_target*: a temporary directory; can be deleted
- *./target_dir/model_input.csv*: a temporary file; can be deleted
- *./target_dir/target.json*: sentences with odds ratio tuples

In *./target_dir/target.json*:

- The model extracts 5-tuples of (variant, disease, odds raito, confidence interval, p-value)
- Missing slot values are indicated by **x**
- With **--complete_triplet_only**, a tuple is extracted only if its variant, disease, and odds raito values are not missing

Sample output:

```json
[
    ...
    {
        "pmid": "25279164",
        "sent_id": 5,
        "sentence": "The results indicated that the GIGYF2 C.3630A>G polymorphism increased the risk of PD by 37% [P=0.008; odds ratio (OR), 1.37; 95% confidence interval (CI), 1.08-1.73] and that the GIGYF2 C.167G>A polymorphism was significantly associated with PD (P=0.003; OR, 3.67; 95% CI, 1.56-8.68).",
        "span_list": [[0, 3], [4, 11], [12, 21], [22, 26], [27, 30], [31, 37], [38, 45], [45, 46], [46, 47], [48, 60], [61, 70], [71, 74], [75, 79], [80, 82], [83, 85], [86, 88], [89, 91], [91, 92], [93, 94], [94, 101], [101, 102], [103, 107], [108, 113], [114, 115], [115, 117], [117, 118], [118, 119], [120, 124], [124, 125], [126, 128], [128, 129], [130, 140], [141, 149], [150, 151], [151, 153], [153, 154], [154, 155], [156, 165], [165, 166], [167, 170], [171, 175], [176, 179], [180, 186], [187, 193], [193, 194], [194, 195], [196, 208], [209, 212], [213, 226], [227, 237], [238, 242], [243, 245], [246, 247], [247, 254], [254, 255], [256, 258], [258, 259], [260, 264], [264, 265], [266, 268], [268, 269], [270, 272], [272, 273], [274, 283], [283, 284], [284, 285]],
        "token_list": ["The", "results", "indicated", "that", "the", "GIGYF2", "C.3630A", ">", "G", "polymorphism", "increased", "the", "risk", "of", "PD", "by", "37", "%", "[", "P=0.008", ";", "odds", "ratio", "(", "OR", ")", ",", "1.37", ";", "95", "%", "confidence", "interval", "(", "CI", ")", ",", "1.08-1.73", "]", "and", "that", "the", "GIGYF2", "C.167G", ">", "A", "polymorphism", "was", "significantly", "associated", "with", "PD", "(", "P=0.003", ";", "OR", ",", "3.67", ";", "95", "%", "CI", ",", "1.56-8.68", ")", "."],
        "mention_list": [
            {"name": "GIGYF2", "type": "gene", "pos": [5, 6], "real_pos": [31, 37]},
            {"name": "C.3630A>G", "type": "mutation", "pos": [6, 9], "real_pos": [38, 47]},
            {"name": "PD", "type": "disease", "pos": [14, 15], "real_pos": [83, 85]},
            {"name": "GIGYF2", "type": "gene", "pos": [42, 43], "real_pos": [180, 186]},
            {"name": "C.167G>A", "type": "mutation", "pos": [43, 46], "real_pos": [187, 195]},
            {"name": "PD", "type": "disease", "pos": [51, 52], "real_pos": [243, 245]}
        ],
        "var_dis_or_ci_pv": [
            ["C.3630A>G", "PD", "1.37", "1.08-1.73", "0.008"],
            ["C.167G>A", "PD", "3.67", "1.56-8.68", "0.003"]
        ]
    },
    ...
]
```

### 5. Speed Profile

Environment:
- RTX 2080ti (x1)

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
- 35 sec
