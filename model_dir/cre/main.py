import os
import csv
import sys
import copy
import json
import logging
import argparse
import subprocess
from collections import defaultdict

import numpy as np
from transformers import BertTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
csv.field_size_limit(sys.maxsize)
csv.register_dialect(
    "csv", delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True,
    escapechar=None, lineterminator="\n", skipinitialspace=False,
)
csv.register_dialect(
    "tsv", delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, doublequote=False,
    escapechar=None, lineterminator="\n", skipinitialspace=False,
)
"""
raw datum format:
{
    "pmid": "1",
    "sent_id": 0,
    "sentence": "I ate a cat.",
    "span_list": [(0,1), (2,5), (6,7), (8,11), (11,12)],
    "token_list": ["I", "ate", "a", "cat", "."],
    "mention_list": [
        {"name": "cat", "type": "drug", "pos": [3,4], "real_pos": [8,11], "id": [["BERN:123", "MESH:456"]]},
    ],
}
# span_list could be None if post-hoc matching of the sentence and its token sequence failed
# real_pos could be [-1,-1] if matching of the sentence and the mention failed
"""


def read_lines(file, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        line_list = f.read().splitlines()

    if write_log:
        lines = len(line_list)
        logger.info(f"Read {lines:,} lines")
    return line_list


def read_json(file, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    if write_log:
        objects = len(data)
        logger.info(f"Read {objects:,} objects")
    return data


def write_json(file, data, indent=None, write_log=True):
    if write_log:
        objects = len(data)
        logger.info(f"Writing {objects:,} objects")

    with open(file, "w", encoding="utf8") as f:
        json.dump(data, f, indent=indent)

    if write_log:
        logger.info(f"Written to {file}")
    return data


def read_csv(file, dialect, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f, dialect=dialect)
        row_list = [row for row in reader]

    if write_log:
        rows = len(row_list)
        logger.info(f"Read {rows:,} rows")
    return row_list


def write_csv(file, dialect, row_list, write_log=True):
    if write_log:
        rows = len(row_list)
        logger.info(f"Writing {rows:,} rows")

    with open(file, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, dialect=dialect)
        for row in row_list:
            writer.writerow(row)

    if write_log:
        logger.info(f"Written to {file}")
    return


def get_vd_mention_group(mention_list, variant_tag, disease_tag):
    # Found mentions for each id; a mention might have multiple id lists
    # mention name is used when there is no id list for a mention
    v_eid_mset = defaultdict(lambda: set())
    d_eid_mset = defaultdict(lambda: set())
    for mi, mention in enumerate(mention_list):
        eid_group_list = mention["id"] if mention["id"] else [[mention["name"]]]
        if mention["type"] == variant_tag:
            for eid_group in eid_group_list:
                for eid in eid_group:
                    v_eid_mset[eid].add(mi)
        elif mention["type"] == disease_tag:
            for eid_group in eid_group_list:
                for eid in eid_group:
                    d_eid_mset[eid].add(mi)

    # Merge mention groups, i.e. when different id have the same set of mentions
    v_msetstring_set = set()
    for _, mset in v_eid_mset.items():
        msetstring = "-".join(str(mi) for mi in sorted(mset))
        v_msetstring_set.add(msetstring)
    d_msetstring_set = set()
    for _, mset in d_eid_mset.items():
        msetstring = "-".join(str(mi) for mi in sorted(mset))
        d_msetstring_set.add(msetstring)

    # Let each group contain both string and list formats: tuple(1-2-3, [1,2,3])
    v_group_list = [
        (v_mset_string, [int(mi) for mi in v_mset_string.split("-")])
        for v_mset_string in sorted(v_msetstring_set)
    ]
    d_group_list = [
        (d_mset_string, [int(mi) for mi in d_mset_string.split("-")])
        for d_mset_string in sorted(d_msetstring_set)
    ]

    return v_group_list, d_group_list


def create_model_input(source_file, meta_file, input_file, variant_tag, disease_tag):
    data = read_json(source_file)
    extracted_meta_data = [["pmid", "sent_id", "variant_id", "disease_id"]]
    extracted_sentence_data = []
    piecer = BertTokenizer("./rbert_relation_extraction/model/biobert/vocab.txt")

    for datum in data:
        pmid = datum["pmid"]
        sent_id = datum["sent_id"]
        mention_list = datum["mention_list"]
        token_list = datum["token_list"]
        v_group_list, d_group_list = get_vd_mention_group(mention_list, variant_tag, disease_tag)

        # mask sentence
        for mention in mention_list:
            if mention["type"] == variant_tag:
                mask = "VARIANT"
            elif mention["type"] == disease_tag:
                mask = "DISEASE"
            else:
                continue
            ml, mr = mention["pos"]
            token_list[ml] = mask
            for i in range(ml + 1, mr):
                token_list[i] = None

        # create an input sentence per variant-disease pair
        for v_string, v_mi_list in v_group_list:
            for d_string, d_mi_list in d_group_list:
                sentence = copy.deepcopy(token_list)

                for mi in v_mi_list:
                    mention = mention_list[mi]
                    sentence[mention["pos"][0]] = "<e1> VARIANT </e1>"
                for mi in d_mi_list:
                    mention = mention_list[mi]
                    sentence[mention["pos"][0]] = "<e2> DISEASE </e2>"
                sentence = [token for token in sentence if token]
                sentence = " ".join(sentence)

                # The model cannot handle extremely long sentences
                piece_list = piecer.tokenize(sentence)
                if len(piece_list) > 370:
                    continue

                extracted_meta_data.append([pmid, sent_id, v_string, d_string])
                extracted_sentence_data.append([sentence])

    if len(extracted_meta_data) == 1:
        extracted_meta_data.append(["x", 0, "x", "x"])
        extracted_sentence_data.append(["<e1> VARIANT </e1> ate a <e2> DISEASE </e2>"])
    write_csv(meta_file, "csv", extracted_meta_data)
    write_csv(input_file, "tsv", extracted_sentence_data)
    return


def run_model(source_file, output_file):
    logger.info("run_model()")

    relative_source = source_file if source_file.startswith("/") else os.path.join("..", source_file)
    relative_output = output_file if output_file.startswith("/") else os.path.join("..", output_file)

    subprocess.run(
        [
            "python", "predict.py",
            "--model", "./model/biobert-nonanndisge3",
            "--input_file", relative_source,
            "--output_file", relative_output,
        ],
        cwd="rbert_relation_extraction",
    )

    logger.info(f"Saved to {output_file}")
    return


def collect_result(source_file, meta_file, output_file, target_file, indent):
    source_data = read_json(source_file)

    meta_data = read_csv(meta_file, "csv")[1:]
    metas = len(meta_data)

    output_data = read_lines(output_file)
    lines = len(output_data)
    assert metas == lines

    score_data = np.load(output_file + ".npy", allow_pickle=True)
    n, d = score_data.shape
    assert n == metas
    assert d == 4

    label_list = ["other", "cause-positive", "appositive", "in-patient"]
    label_map = {
        "Other": 0,
        "Vpositive-D(e1,e2)": 1,
        "Vappositive-D(e1,e2)": 2,
        "Vpatient-D(e1,e2)": 3,
    }

    ps_to_relation = defaultdict(lambda: [])
    for meta_i in range(metas):
        pmid, sent_id, v_string, d_string = meta_data[meta_i]
        if pmid == "x":
            continue
        sent_id = int(sent_id)
        v_mi_list = [int(mi) for mi in v_string.split("-")]
        d_mi_list = [int(mi) for mi in d_string.split("-")]

        label_index = label_map[output_data[meta_i]]
        score = score_data[meta_i, :]
        assert max(range(len(score)), key=lambda i: score[i]) == label_index
        score = score - min(score)
        score = score / sum(score)
        score = [
            (label_list[i], f"{s:.1%}")
            for i, s in enumerate(score)
        ]

        ps_to_relation[(pmid, sent_id)].append([v_mi_list, d_mi_list, label_index, score])

    positive_sentences = 0
    sentences = 0
    triplets = 0

    for datum in source_data:
        pmid = datum["pmid"]
        sent_id = datum["sent_id"]
        mention_list = datum["mention_list"]
        triplet_list = []
        positive = False

        for v_mi_list, d_mi_list, label_index, score in ps_to_relation[(pmid, sent_id)]:
            v_name = mention_list[v_mi_list[0]]["name"]
            d_name = mention_list[d_mi_list[0]]["name"]
            label = label_list[label_index]
            if label != "other":
                positive = True
            triplet_list.append({
                "h_mention": v_mi_list,
                "t_mention": d_mi_list,
                "score": score,
                "triplet": [v_name, label, d_name],
            })

        datum["triplet_list"] = triplet_list
        if triplet_list:
            triplets += len(triplet_list)
            sentences += 1
            if positive:
                positive_sentences += 1

    logger.info(f"{positive_sentences:,} sentences with positive triplets")
    logger.info(f"{sentences:,} sentences with triplets")
    logger.info(f"{triplets:,} triplets")
    write_json(target_file, source_data, indent=indent)
    return


def run_closed_relation_extraction(arg):
    os.makedirs(arg.target_dir, exist_ok=True)
    meta_file = os.path.join(arg.target_dir, "meta.csv")
    input_file = os.path.join(arg.target_dir, "input.txt")
    output_file = os.path.join(arg.target_dir, "output.txt")
    target_file = os.path.join(arg.target_dir, "target.json")
    indent = arg.indent if arg.indent >= 0 else None

    create_model_input(arg.source_file, meta_file, input_file, arg.variant_tag, arg.disease_tag)
    run_model(input_file, output_file)
    collect_result(arg.source_file, meta_file, output_file, target_file, indent)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="./source.json")
    parser.add_argument("--target_dir", type=str, default="./target_dir")

    parser.add_argument("--variant_tag", type=str, default="mutation")
    parser.add_argument("--disease_tag", type=str, default="disease")

    parser.add_argument("--indent", type=int, default=2)

    arg = parser.parse_args()
    run_closed_relation_extraction(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
