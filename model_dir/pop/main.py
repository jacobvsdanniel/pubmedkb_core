import os
import re
import csv
import sys
import json
import logging
import argparse
import subprocess

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


def create_model_input(source_file, number_measure_file, species_file, model_input_file):
    window = [20, 50]
    number = re.compile(r"(([0-9]+[, ])*[0-9]+[ ]?/[ ]?)?([0-9]+[, ])*[0-9]+")  # "123 456 789 / 123,456,789"
    keyword_set = set()

    line_list = read_lines(number_measure_file)
    for line in line_list:
        if line:
            keyword_set.add(line)
    line_list = read_lines(species_file)
    for line in line_list:
        if line:
            keyword_set.add(line)
    keywords = len(keyword_set)
    logger.info(f"keywords: {sorted(keyword_set)}")
    logger.info(f"{keywords} keywords")

    data = read_json(source_file)
    extracted_data = [["pmid", "sent_id", "sentence", "table"]]

    for datum in data:
        pmid = datum["pmid"]
        sent_id = datum["sent_id"]
        sentence = datum["sentence"]
        table = ";"
        key_phrase_found = False

        for match in number.finditer(sentence):
            li, ri = match.start(), match.end()
            if (li > 0 and sentence[li - 1] != " ") or (ri < len(sentence) and sentence[ri] != " "):
                continue
            li = max(0, li - window[0])
            ri = min(len(sentence), ri + window[1])
            phrase = sentence[li:ri].lower()
            for keyword in keyword_set:
                if keyword in phrase:
                    key_phrase_found = True
                    break
            else:
                continue
            break
        if key_phrase_found:
            extracted_data.append([pmid, sent_id, sentence, table])

    if len(extracted_data) == 1:
        extracted_data.append(["x", 0, "I ate a cat.", ";"])
    write_csv(model_input_file, "csv", extracted_data)
    return


def run_model(model_input_file, model_output_file, model_dir, target_dir):
    logger.info("run_model()")
    subprocess.run([
        "python", "run_model.py",
        "--model_name_or_path", model_dir,
        "--output_dir", target_dir,
        "--train_file", os.path.join(model_dir, "train.csv"),
        "--validation_file", os.path.join(model_dir, "dev.csv"),
        "--test_file", model_input_file,
        "--prediction_file", model_output_file,
        "--text_column", "sentence",
        "--summary_column", "table",
        "--do_predict",
        "--source_prefix", "summarize: ",
        "--max_source_length", "2500",
        "--max_target_length", "500",
        "--per_device_eval_batch_size=8",
        "--overwrite_cache", "True",
        "--predict_with_generate",
    ])
    logger.info(f"Saved to {model_output_file}")
    return


def parse_model_output(source_file, model_input_file, model_output_file, target_file, indent):
    data = read_json(source_file)
    sentences = 0
    populations = 0

    input_data = read_csv(model_input_file, "csv")[1:]
    output_data = read_lines(model_output_file)
    assert len(input_data) == len(output_data)

    ps_to_table = {}

    for model_input, model_output in zip(input_data, output_data):
        pmid, sent_id, _, _ = model_input
        if pmid == "x":
            continue
        sent_id = int(sent_id)

        table = []
        for phrase in model_output.split(";"):
            phrase = phrase.strip()
            if phrase:
                table.append(phrase)

        if table:
            ps_to_table[(pmid, sent_id)] = table

    for datum in data:
        pmid = datum["pmid"]
        sent_id = datum["sent_id"]
        table = ps_to_table.get((pmid, sent_id), [])
        datum["population"] = table

        if table:
            sentences += 1
            populations += len(table)

    logger.info(f"{sentences:,} sentences with populations")
    logger.info(f"{populations:,} populations")
    write_json(target_file, data, indent=indent)
    return


def run_population_extraction(arg):
    os.makedirs(arg.target_dir, exist_ok=True)
    model_input_file = os.path.join(arg.target_dir, "model_input.csv")
    model_output_file = os.path.join(arg.target_dir, "model_output.txt")
    target_file = os.path.join(arg.target_dir, "target.json")
    indent = arg.indent if arg.indent >= 0 else None

    create_model_input(arg.source_file, arg.number_measure_file, arg.species_file, model_input_file)
    run_model(model_input_file, model_output_file, arg.model_dir, arg.target_dir)
    parse_model_output(arg.source_file, model_input_file, model_output_file, target_file, indent)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="./source.json")
    parser.add_argument("--target_dir", type=str, default="./target_dir")
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument("--number_measure_file", type=str, default="./number_measure.txt")
    parser.add_argument("--species_file", type=str, default="./species.txt")
    parser.add_argument("--indent", type=int, default=2)

    arg = parser.parse_args()
    run_population_extraction(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
