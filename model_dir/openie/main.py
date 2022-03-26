import os
import re
import sys
import json
import html
import math
import logging
import argparse
import subprocess
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
max_sentence_length = 500
"""
raw datum format:
{
    "pmid": "1",
    "sent_id": 0,
    "sentence": "I ate a cat.",
    "span_list": [(0,1), (2,5), (6,7), (8,11), (11,12)],
    "token_list": ["I", "ate", "a", "cat", "."],
    "mention_list": [
        {"name": "cat", "type": "species", "pos": [3,4], "real_pos": [8,11]},
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


def create_batch_data(source_file, all_source_dir, all_target_dir, indent, batch_size):
    data = read_json(source_file)
    data = [
        datum
        for datum in data
        if len(datum["mention_list"]) >= 2
    ]
    sentences = len(data)

    os.makedirs(all_source_dir, exist_ok=True)
    os.makedirs(all_target_dir, exist_ok=True)

    for si in range(0, sentences, batch_size):
        start, end = si, min(si + batch_size, sentences)

        batch_source_file = os.path.join(all_source_dir, f"{start}_{end}.json")
        write_json(batch_source_file, data[start:end], indent=indent)

        batch_target_dir = os.path.join(all_target_dir, f"{start}_{end}")
        os.makedirs(batch_target_dir, exist_ok=True)

    return sentences


def add_mask_data(data):
    logger.info("add_mask_data()")
    for datum in data:
        masked_sentence = [c for c in datum["sentence"]]
        for mi, mention in enumerate(datum["mention_list"]):
            ci, cj = mention["real_pos"]
            if ci == -1:
                continue
            masked_sentence[ci] = f"ENTITY{mi}X"
            for i in range(ci + 1, cj):
                masked_sentence[i] = None
        masked_sentence = [
            c
            for c in masked_sentence[:max_sentence_length]
            if c
        ]
        masked_sentence = "".join(masked_sentence)
        datum["masked_sentence"] = masked_sentence
    return


def extract_delimited_sentence(data, sentence_file_prefix, batch_processes):
    logger.info("extract_delimited_sentence()")

    sentences = len(data)
    sub_sentences = math.ceil(sentences / batch_processes)
    delimiter = "I ate a cat."

    for pi in range(batch_processes):
        sentence_file = f"{sentence_file_prefix}_{pi}.txt"
        si_l = pi * sub_sentences
        si_r = min((pi + 1) * sub_sentences, sentences)

        with open(sentence_file, "w") as f:
            for di in range(si_l, si_r):
                masked_sentence = data[di]["masked_sentence"]
                masked_sentence = html.escape(masked_sentence, quote=False)
                f.write(f"{masked_sentence}\n")
                f.write(f"{delimiter}\n")
    return


def extract_delimited_triplet(
        sentence_file_prefix, triplet_file, jar_dir, file_list_file_prefix,
        batch_processes, process_memory,
):
    logger.info("extract_delimited_triplet()")
    process_list = []
    result = ""

    # Spawn subprocesses to run java
    for pi in range(batch_processes):
        sentence_file = f"{sentence_file_prefix}_{pi}.txt"
        file_list_file = f"{file_list_file_prefix}_{pi}.txt"

        with open(file_list_file, "w") as f:
            f.write(sentence_file)
        process = subprocess.Popen(
            ["sh", "openie.sh", jar_dir, file_list_file, process_memory],
            stdout=subprocess.PIPE,
        )
        process_list.append(process)

    # Wait and collect results from each subprocess
    for pi, process in enumerate(process_list):
        stdout, _ = process.communicate()
        stdout = stdout.decode("utf8")
        if not stdout:
            continue
        if stdout[-1] != "\n":
            stdout += "\n"
        result += stdout

    with open(triplet_file, "w") as f:
        f.write(result)
    return


def add_triplet_data(data, delimited_triplet_file):
    logger.info("add_triplet_data()")

    line_list = read_lines(delimited_triplet_file)
    kb_list = []
    kb = []  # A list of triplets
    for line in line_list:
        line = line[8:-1]
        if line == "I; ate; cat":
            kb_list.append(kb)
            kb = []
        else:
            line = line.split("; ")
            assert len(line) == 3
            kb.append(line)
    assert len(kb_list) == len(data)

    for si, kb in enumerate(kb_list):
        data[si]["triplet_list"] = kb
    return


def is_subsequence(t1, t2):
    it = iter(t2)
    return all(any(c2 == c1 for c2 in it) for c1 in t1)


def get_reduced_triplet(perfect_triplet_list):
    perfect_triplet_text_list = [
        (perfect, triplet, " ".join(triplet))
        for perfect, triplet in perfect_triplet_list
    ]
    perfect_triplet_text_list = sorted(set(perfect_triplet_text_list))

    reduced_list = []
    for i1, ptt1 in enumerate(perfect_triplet_text_list):
        redundant = False
        for i2, ptt2 in enumerate(perfect_triplet_text_list):
            if i1 == i2:
                continue
            # A triplet is redundant if it is [super-sequence but less perfect] or [sub-sequence and not more perfect]
            text1, text2 = ptt1[2], ptt2[2]
            is_sub = is_subsequence(text1, text2)
            is_sup = is_subsequence(text2, text1)
            perfect1, perfect2 = ptt1[0], ptt2[0]
            if is_sup and (not perfect1 and perfect2):
                redundant = True
            elif is_sub and (not perfect1 or perfect2):
                redundant = True
        if not redundant:
            reduced_list.append((ptt1[0], ptt1[1]))
    return reduced_list


def add_matched_triplet_data(data):
    logger.info("add_matched_triplet_data()")
    mention_expression = re.compile(r"ENTITY\d+X")

    for datum in data:
        mention_list = datum["mention_list"]
        triplet_list = datum["triplet_list"]
        ht_to_triplet_list = defaultdict(lambda: [])

        # Match (head, tail) to mention pair
        for triplet in triplet_list:
            h, r, t = triplet
            h_match_list = mention_expression.findall(h)
            r_match_list = mention_expression.findall(r)
            t_match_list = mention_expression.findall(t)
            if len(h_match_list) != 1 or len(r_match_list) != 0 or len(t_match_list) != 1:
                continue
            h_match = h_match_list[0]
            t_match = t_match_list[0]
            perfect_match = h_match == h and t_match == t
            h_match = int(h_match[6:-1])
            t_match = int(t_match[6:-1])
            if h_match == t_match:
                continue
            ht_to_triplet_list[(h_match, t_match)].append(
                (perfect_match, tuple(triplet))
            )

        # Remove redundant triplets per mention pair
        triplet_list = []
        for (h_mi, t_mi), perfect_triplet_list in ht_to_triplet_list.items():
            perfect_triplet_list = get_reduced_triplet(perfect_triplet_list)

            for perfect, triplet in perfect_triplet_list:
                h_name = mention_list[h_mi]["name"]
                t_name = mention_list[t_mi]["name"]
                h, r, t = triplet
                h = mention_expression.sub(lambda _: h_name, h)
                t = mention_expression.sub(lambda _: t_name, t)
                triplet = (h, r, t)

                triplet_list.append(
                    {
                        "h_mention": h_mi,
                        "t_mention": t_mi,
                        "perfect_match": perfect,
                        "triplet": triplet,
                    }
                )

        datum["triplet_list"] = triplet_list
    return


def run_extraction(arg, source_file, target_dir, indent):
    sentence_file_prefix = os.path.join(target_dir, "sentence")
    triplet_file = os.path.join(target_dir, "triplet.txt")
    file_list_file_prefix = os.path.join(target_dir, "file_list")
    target_file = os.path.join(target_dir, "target.json")

    data = read_json(source_file)

    if not arg.no_preprocess:
        add_mask_data(data)

    if not arg.no_openie:
        extract_delimited_sentence(data, sentence_file_prefix, arg.batch_processes)
        extract_delimited_triplet(
            sentence_file_prefix, triplet_file, arg.jar_dir, file_list_file_prefix,
            arg.batch_processes, arg.process_memory,
        )

    if not arg.no_postprocess:
        add_triplet_data(data, triplet_file)
        add_matched_triplet_data(data)
        data = [datum for datum in data if datum["triplet_list"]]

    write_json(target_file, data, indent=indent)
    return


def collect_batch_result(source_file, all_target_dir, target_dir, indent):
    ps_to_triplet_list = {}
    batch_target_dir_list = sorted(os.listdir(all_target_dir), key=lambda n: int(n.split("_")[0]))

    for batch_target_dir in batch_target_dir_list:
        batch_target_file = os.path.join(all_target_dir, batch_target_dir, "target.json")
        if not os.path.exists(batch_target_file):
            continue
        batch = read_json(batch_target_file)
        for datum in batch:
            pmid = datum["pmid"]
            sent_id = datum["sent_id"]
            triplet_list = datum["triplet_list"]
            ps_to_triplet_list[(pmid, sent_id)] = triplet_list

    data = read_json(source_file)
    sentences = 0
    triplets = 0
    perfect_triplets = 0

    for datum in data:
        pmid = datum["pmid"]
        sent_id = datum["sent_id"]
        triplet_list = ps_to_triplet_list.get((pmid, sent_id), [])
        datum["triplet_list"] = triplet_list

        if triplet_list:
            sentences += 1
            triplets += len(triplet_list)
            for triplet in triplet_list:
                if triplet["perfect_match"]:
                    perfect_triplets += 1

    logger.info(f"{sentences:,} sentences with triplets")
    logger.info(f"{triplets:,} triplets")
    logger.info(f"{perfect_triplets:,} perfect triplets")

    target_file = os.path.join(target_dir, "target.json")
    write_json(target_file, data, indent=indent)
    return


def run_batch_extraction(arg):
    os.makedirs(arg.target_dir, exist_ok=True)
    all_source_dir = os.path.join(arg.target_dir, "all_source")
    all_target_dir = os.path.join(arg.target_dir, "all_target")
    indent = arg.indent if arg.indent >= 0 else None

    sentences = create_batch_data(arg.source_file, all_source_dir, all_target_dir, indent, arg.batch_size)

    for si in range(0, sentences, arg.batch_size):
        start, end = si, min(si + arg.batch_size, sentences)
        logger.info(f"Running [{start}, {end}]")
        batch_source_file = os.path.join(all_source_dir, f"{start}_{end}.json")
        batch_target_dir = os.path.join(all_target_dir, f"{start}_{end}")
        run_extraction(arg, batch_source_file, batch_target_dir, indent)

    collect_batch_result(arg.source_file, all_target_dir, arg.target_dir, indent)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="./source.json")
    parser.add_argument("--target_dir", type=str, default="./target_dir")
    parser.add_argument("--jar_dir", type=str, default="./jar_dir/")
    parser.add_argument("--indent", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--batch_processes", type=int, default=4)
    parser.add_argument("--process_memory", type=str, default="6g")

    parser.add_argument("--no_preprocess", action="store_true")
    parser.add_argument("--no_openie", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")

    arg = parser.parse_args()
    run_batch_extraction(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
