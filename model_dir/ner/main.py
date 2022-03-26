import os
import csv
import sys
import json
import logging
import argparse
import subprocess

from pytorch_pretrained_bert import BertTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

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
max_sentence_pieces = 500
"""
raw datum format:
{
    "pmid": "1",
    "sent_id": 0,
    "sentence": "I ate a cat.",
    "span_list": [(0,1), (2,5), (6,7), (8,11), (11,12)],
    "token_list": ["I", "ate", "a", "cat", "."],
}
# span_list could be None if post-hoc matching of the sentence and its token sequence failed
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


def split_token_list(token_list, piecer):
    tokenlist_list = []
    sub_token_list = []
    pieces = 0

    for ti, token in enumerate(token_list):
        token_pieces = len(piecer.tokenize(token))

        if pieces + token_pieces < max_sentence_pieces:
            sub_token_list.append(token)
            pieces += token_pieces
            continue

        tokenlist_list.append(sub_token_list)
        sub_token_list = [token]
        pieces = token_pieces

    if sub_token_list:
        tokenlist_list.append(sub_token_list)
    return tokenlist_list


def create_model_input(source_file, input_file):
    data = read_json(source_file)
    piecer = BertTokenizer("ner/model_dir/vocab.txt", do_lower_case=False)
    extracted_data = []
    split_sentences = 0

    for datum in data:
        token_list = [
            token[:max_sentence_pieces]
            for token in datum["token_list"]
        ]

        if len(datum["sentence"]) < max_sentence_pieces:
            tokenlist_list = [token_list]
        else:
            tokenlist_list = split_token_list(token_list, piecer)
        split_sentences += len(tokenlist_list)

        for token_list in tokenlist_list:
            for token in token_list:
                extracted_data.append([token, "O"])
            extracted_data.append([])
    logger.info(f"{split_sentences:,} split_sentences")

    write_csv(input_file, "tsv", extracted_data)
    return


def run_model(source_file, target_dir):
    logger.info("Running model")
    if not source_file.startswith("/"):
        source_file = os.path.join("..", source_file)
    if not target_dir.startswith("/"):
        target_dir = os.path.join("..", target_dir)

    subprocess.run(
        [
            "python", "main.py",
            "--source_file", source_file,
            "--target_dir", target_dir,
            "--model_dir", "model_dir",
            "--device", "cuda:0",
            "--batch_size", "32",
        ],
        cwd="ner",
    )
    logger.info(f"Model finished")
    return


def get_mention_list(tag_list):
    mention_list = []
    begin_ti = None
    begin_mention_label = None

    for ti, tag in enumerate(tag_list):
        seq_label = tag[0]
        mention_label = None if seq_label == "O" else tag[2:]

        if seq_label == "B":
            if begin_ti:
                mention_list.append({
                    "pos": [begin_ti, ti],
                    "type": begin_mention_label,
                })
            begin_ti = ti
            begin_mention_label = mention_label

        elif seq_label == "O":
            if begin_ti:
                mention_list.append({
                    "pos": [begin_ti, ti],
                    "type": begin_mention_label,
                })
            begin_ti = None
            begin_mention_label = None

        elif begin_ti and not mention_label == begin_mention_label:
            begin_ti = None
            begin_mention_label = None

    if begin_ti is not None:
        mention_list.append({
            "pos": [begin_ti, len(tag_list)],
            "type": begin_mention_label,
        })
    return mention_list


def get_character_position(sentence, token_list, ti, tj):
    no_space_prefix = "".join(token_list[:ti])
    first_name = token_list[ti]
    no_space_name = "".join(token_list[ti:tj])
    ci = len(no_space_prefix) - 1

    while True:
        ci = sentence.find(first_name, ci + 1)
        if ci == -1:
            break

        cj = ci
        offset = 0

        while cj < len(sentence) and offset < len(no_space_name):
            if sentence[cj] == " ":
                cj += 1
            elif sentence[cj] == no_space_name[offset]:
                cj += 1
                offset += 1
            else:
                break

        if offset == len(no_space_name):
            return ci, cj

    return -1, -1


def get_named_mention_list(sentence, span_list, token_list, mention_list, detokenizer):
    named_mention_list = []

    for mention in mention_list:
        _type = mention["type"]
        ti, tj = mention["pos"]  # token position

        # find character position
        # if all methods fail, (ci, cj) == (-1, -1)
        if span_list is not None:
            # method 1: use token_position-to-character_position mapping
            ci, cj = span_list[ti][0], span_list[tj - 1][1]
            name = sentence[ci:cj]
        else:
            # method 2: find the token sequence in the sentence
            ci, cj = get_character_position(sentence, token_list, ti, tj)
            if ci != -1:
                name = sentence[ci:cj]
            else:
                # method 3: find the detokenized token sequence in the sentence
                name = detokenizer.detokenize(token_list[ti:tj])
                ci = sentence.find(name)
                if ci != -1:
                    cj = ci + len(name)

        named_mention = {
            "name": name,
            "type": _type,
            "pos": (ti, tj),
            "real_pos": (ci, cj),
        }
        named_mention_list.append(named_mention)

    return named_mention_list


def collect_result(source_file, output_file, target_file, indent):
    source_data = read_json(source_file)
    output_data = read_csv(output_file, "tsv")

    di = 0
    mentions = 0
    token_list = []
    tag_list = []
    detokenizer = TreebankWordDetokenizer()

    logger.info("Collecting mentions")

    for row in output_data:
        if row:
            token_list.append(row[0])
            tag_list.append(row[2])
            continue
        if len(token_list) < len(source_data[di]["token_list"]):
            continue
        assert token_list == [t[:max_sentence_pieces] for t in source_data[di]["token_list"]]
        mention_list = get_mention_list(tag_list)

        sentence = source_data[di]["sentence"]
        span_list = source_data[di]["span_list"]
        token_list = source_data[di]["token_list"]
        mention_list = get_named_mention_list(sentence, span_list, token_list, mention_list, detokenizer)

        source_data[di]["mention_list"] = mention_list
        di += 1
        mentions += len(mention_list)
        token_list = []
        tag_list = []

    assert not token_list
    assert not tag_list
    assert di == len(source_data)

    logger.info(f"Collected {mentions:,} mentions")
    write_json(target_file, source_data, indent=indent)
    return


def run_ner(arg):
    os.makedirs(arg.target_dir, exist_ok=True)
    input_file = os.path.join(arg.target_dir, "input.txt")
    output_file = os.path.join(arg.target_dir, "target.tsv")
    target_file = os.path.join(arg.target_dir, "target.json")
    indent = arg.indent if arg.indent >= 0 else None

    create_model_input(arg.source_file, input_file)
    run_model(input_file, arg.target_dir)
    collect_result(arg.source_file, output_file, target_file, indent)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="./source.json")
    parser.add_argument("--target_dir", type=str, default="./target_dir")
    parser.add_argument("--indent", type=int, default=2)
    arg = parser.parse_args()

    run_ner(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
