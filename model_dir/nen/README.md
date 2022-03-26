# NEN Tool

## Development Environment

- The same as *./server*

*Note: That repo does NOT require CUDA (running on CPU only).*

## How to Run

### 1. Prepare models

- Follow the instructions in *./server*

### 2. Prepare Source Data

Possible NER types:
- gene (ignoring case, output will become lowercased)
- mutation (ignoring case, output will become lowercased)
- disease (ignoring case, output will become lowercased)
- drug (ignoring case, output will become lowercased)
- VARIANT (case-sentitive, output will become *mutation*)
- Chemical (case-sentitive, output will become *drug*)

An example input file is available at ./source.json

```json
[
    ...
    {
        "pmid":"7981223",
        "sent_id": 2,
        "sentence": "We measured the kinetics of unfolding and refolding of two reduced and carboxymethylated variants of ribonuclease T1 with one cis proline (the Ser54Gly/Pro55Asn variant) and with two cis prolines (the wild-type protein) as a function of the NaCl concentration.",
        "span_list": [[0, 2], [3, 11], [12, 15], [16, 24], [25, 27], [28, 37], [38, 41], [42, 51], [52, 54], [55, 58], [59, 66], [67, 70], [71, 88], [89, 97], [98, 100], [101, 113], [114, 116], [117, 121], [122, 125], [126, 129], [130, 137], [138, 139], [139, 142], [143, 160], [161, 168], [168, 169], [170, 173], [174, 178], [179, 182], [183, 186], [187, 195], [196, 197], [197, 200], [201, 210], [211, 218], [218, 219], [220, 222], [223, 224], [225, 233], [234, 236], [237, 240], [241, 245], [246, 259], [259, 260]],
        "token_list": ["We", "measured", "the", "kinetics", "of", "unfolding", "and", "refolding", "of", "two", "reduced", "and", "carboxymethylated", "variants", "of", "ribonuclease", "T1", "with", "one", "cis", "proline", "(", "the", "Ser54Gly/Pro55Asn", "variant", ")", "and", "with", "two", "cis", "prolines", "(", "the", "wild-type", "protein", ")", "as", "a", "function", "of", "the", "NaCl", "concentration", "."],
        "mention_list": [
            {"name": "Ser54Gly/Pro55Asn", "type": "VARIANT", "pos": [23, 24], "real_pos": [143, 160]},
            {"name": "NaCl", "type": "Chemical", "pos": [41, 42], "real_pos": [241, 245]}
        ]
    },
    ...
]
```

### 3. Run

See *./demo.sh*

It takes about 38 minutes to run the demo.

```bash
python main.py \
--source_file source.json \
--target_file target.json \
--batch_size 1000 \
--processes 8 \
--port 8888 \
--indent 2
```

### 4. Results

- *target.json*: sentences with NER + NEN annotations

### 5. Result format

Possible ID systems include:
- BERN
- MESH
- CHEBI
- OMIM
- MIM
- HGNC
- Ensembl
- miRBase
- IMGT/GENE-DB

Expected results in *target.json*
```json
[
    ...
    {
        "pmid":"7981223",
        "sent_id": 2,
        "sentence": "We measured the kinetics of unfolding and refolding of two reduced and carboxymethylated variants of ribonuclease T1 with one cis proline (the Ser54Gly/Pro55Asn variant) and with two cis prolines (the wild-type protein) as a function of the NaCl concentration.",
        "span_list": [[0, 2], [3, 11], [12, 15], [16, 24], [25, 27], [28, 37], [38, 41], [42, 51], [52, 54], [55, 58], [59, 66], [67, 70], [71, 88], [89, 97], [98, 100], [101, 113], [114, 116], [117, 121], [122, 125], [126, 129], [130, 137], [138, 139], [139, 142], [143, 160], [161, 168], [168, 169], [170, 173], [174, 178], [179, 182], [183, 186], [187, 195], [196, 197], [197, 200], [201, 210], [211, 218], [218, 219], [220, 222], [223, 224], [225, 233], [234, 236], [237, 240], [241, 245], [246, 259], [259, 260]],
        "token_list": ["We", "measured", "the", "kinetics", "of", "unfolding", "and", "refolding", "of", "two", "reduced", "and", "carboxymethylated", "variants", "of", "ribonuclease", "T1", "with", "one", "cis", "proline", "(", "the", "Ser54Gly/Pro55Asn", "variant", ")", "and", "with", "two", "cis", "prolines", "(", "the", "wild-type", "protein", ")", "as", "a", "function", "of", "the", "NaCl", "concentration", "."],
        "mention_list": [
            {"name": "Ser54Gly/Pro55Asn", "type": "mutation", "pos": [23, 24], "real_pos": [143, 160], "id": [["BERN:1973004"], ["BERN:1973104"]]},
            {"name": "NaCl", "type": "drug", "pos": [41, 42], "real_pos": [241, 245], "id": [["CHEBI:26710", "BERN:314219703"]]}
        ]
    },
    ...
]
```

### 6. Speed Profile

Environment:
- CPU only

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
- 579 sec
