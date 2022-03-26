# NEN Server

Reference: https://github.com/dmis-lab/bern

## Requirements

- Python 3.6 or higher

## Installation

1. Clone this repository and make sure that `./bern` in under `/root/`.
```
|-- root
|   |-- bern
|   `-- readme.md
```

2. Install Java and python packages. 
```bash
apt-get install default-jre
pip3 install -r requirements.txt
```

3. Copy file from `/volume/pubmedkb-covid/NEN` and untar them.
```bash
cp /volume/pubmedkb-covid/NEN/GNormPlusJava.tar.gz ./bern/
cp /volume/pubmedkb-covid/NEN/normalization.tar.gz ./bern/
cp /volume/pubmedkb-covid/NEN/tmVarJava.tar.gz ./bern/
cp /volume/pubmedkb-covid/NEN/pretrainedBERT.tar.gz ./bern/biobert_ner/

cd bern
tar zxvf GNormPlusJava.tar.gz
tar zxvf normalization.tar.gz
tar zxvf tmVarJava.tar.gz

cd biobert_ner
tar zxvf pretrainedBERT.tar.gz
```

## Run
```bash
cd ~/bern
mkdir biobert_ner/tmp
mkdir logs
./run.sh

# Print logs
tail -F logs/nohup_BERN.out

# Pease wait a minute until the log contains:
# [30/Dec/2021 10:58:18.801928] Starting server at http://0.0.0.0:8888
# gid2oid loaded 59849
# goid2goid loaded 3468
# gene meta #ids 42916, #ext_ids 42916
# disease meta #ids 12122, #ext_ids 15040
# chem meta #ids 179063, #ext_ids 179463
# code2mirs size 9447
# mirbase_id2mirna_id size 14945
# mirna_id2accession size 6308
# # of pathway regex 514
```

This script spawns the following background processes:
| Port | Process |
| - | - |
| 8888 | python3 -u server.py --port 8888 |
| 18895 | java -Xmx16G -Xms16G -jar GNormPlusServer.jar |
| 18896 | java -Xmx8G -Xms8G -jar tmVar2Server.jar |
| 18888 | java -Xmx20G -jar gnormplus-normalization_19.jar |
| 18889 | python3 normalizers/species_normalizer.py |
| 18890 | python3 normalizers/chemical_normalizer.py |
| 18891 | python3 normalizers/mutation_normalizer.py |
| 18892 | java -Xmx16G -jar resources/normalizers/disease/disease_normalizer_19.jar |

## How to use
POST address: http://localhost:8888

```python
import requests
import json
body_data = {"param": json.dumps({"text": "N1303K", "type": "mutation"})}
response = requests.post('http://localhost:8888', data=body_data)
result_dict = response.json()
print(result_dict)
```
