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


def get_odds_ratio(text):
    m_list = [
        (m.start(), m.end())
        for m in re.finditer(r"odds ratio", text.lower())
    ]
    m_list += [
        (m.start(), m.end())
        for m in re.finditer(r"[a-zA-Z0-9]*OR[a-zA-Z0-9]*", text)
        if text[m.start():m.end()] in ["OR", "AOR", "aOR"]
    ]
    m_list = sorted(m_list)
    return m_list


def create_model_input(source_file, model_input_file, complete_triplet_only, variant_tag, disease_tag):
    data = read_json(source_file)
    extracted_data = []

    for datum in data:
        pmid = datum["pmid"]
        sent_id = datum["sent_id"]
        mention_list = datum["mention_list"]
        token_list = datum["token_list"]

        # odds ratio must be in the sentence (according to pattern)
        sentence = " ".join(token_list)
        if not get_odds_ratio(sentence):
            continue

        # create a sentence with variant and disease tags
        has_variant = False
        has_disease = False

        for mention in mention_list:
            if mention["type"] == variant_tag:
                tag = "[VARIANT]"
                has_variant = True
            elif mention["type"] == disease_tag:
                tag = "[DISEASE]"
                has_disease = True
            else:
                continue

            ti, tj = mention["pos"]
            token_list[ti] = tag + " " + token_list[ti]
            token_list[tj - 1] = token_list[tj - 1] + " " + tag

        if complete_triplet_only and (not has_variant or not has_disease):
            continue
        sentence = " ".join(token_list)
        table = "[table] no annotation"
        extracted_data.append([pmid, sent_id, sentence, table])

    if len(extracted_data) == 0:
        extracted_data.append(["x", 0, "I ate a cat.", ";"])
    write_csv(model_input_file, "csv", extracted_data)
    return


def create_model_input_ideal(source_file, model_input_file, complete_triplet_only, variant_tag, disease_tag):
    # TODO: re-train a model with original sentence instead of space-joined tokens
    data = read_json(source_file)
    extracted_data = []

    for datum in data:
        pmid = datum["pmid"]
        sent_id = datum["sent_id"]
        mention_list = datum["mention_list"]
        sentence = datum["sentence"]

        # odds ratio must be in the sentence (according to pattern)
        if not get_odds_ratio(sentence):
            continue

        # create a sentence with variant and disease tags
        sentence = [c for c in sentence]
        has_variant = False
        has_disease = False

        for mention in mention_list:
            ci, cj = mention["real_pos"]
            if ci == -1:
                continue

            if mention["type"] == variant_tag:
                tag = "[VARIANT]"
                has_variant = True
            elif mention["type"] == disease_tag:
                tag = "[DISEASE]"
                has_disease = True
            else:
                continue

            sentence[ci] = tag + " " + sentence[ci]
            sentence[cj - 1] = sentence[cj - 1] + " " + tag

        if complete_triplet_only and (not has_variant or not has_disease):
            continue
        sentence = "".join(sentence)
        table = "[table] no annotation"
        extracted_data.append([pmid, sent_id, sentence, table])

    if len(extracted_data) == 0:
        extracted_data.append(["x", 0, "I ate a cat.", ";"])
    write_csv(model_input_file, "csv", extracted_data)
    return


def create_batch_data(source_file, all_source_dir, all_target_dir, batch_size):
    data = read_csv(source_file, "csv")
    sentences = len(data)

    os.makedirs(all_source_dir, exist_ok=True)
    os.makedirs(all_target_dir, exist_ok=True)

    for si in range(0, sentences, batch_size):
        start, end = si, min(si + batch_size, sentences)

        batch_source_file = os.path.join(all_source_dir, f"{start}_{end}.csv")
        batch_data = [["pmid", "sent_id", "sentence", "table"]] + data[start:end]
        write_csv(batch_source_file, "csv", batch_data)

        batch_target_dir = os.path.join(all_target_dir, f"{start}_{end}")
        os.makedirs(batch_target_dir, exist_ok=True)

    return sentences


def run_model(source_file, model_dir, target_dir):
    target_file = os.path.join(target_dir, "target.txt")
    logger.info("run_model()")
    subprocess.run([
        "python", "run_model.py",
        "--model_name_or_path", model_dir,
        "--output_dir", target_dir,
        "--train_file", os.path.join(model_dir, "train.csv"),
        "--validation_file", os.path.join(model_dir, "dev.csv"),
        "--test_file", source_file,
        "--prediction_file", target_file,
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
    logger.info(f"Saved to {target_file}")
    return


def normalize_odds_ratio_prediction(prediction):
    if prediction == "x":
        return "x"
    prediction = prediction.replace(" ", "")
    try:
        if float(prediction) == 0:
            return "x"
    except ValueError:
        return "x"
    return prediction


def normalize_confidence_interval_prediction(prediction):
    if prediction == "x":
        return "x"

    alphabet = "0123456789.-()[]"
    prediction = prediction.replace(" ", "")
    for c in prediction:
        if c not in alphabet:
            return "x"

    v_list = re.findall(r"\d+\.\d+", prediction)
    if len(v_list) != 2:
        return "x"

    return prediction


def normalize_p_value_prediction(prediction):
    if prediction == "x":
        return "x"

    alphabet = "0123456789.-()[]x"
    prediction = prediction.replace(" ", "")
    for c in prediction:
        if c not in alphabet:
            return "x"

    try:
        v = float(prediction)
        if v >= 1 or v == 0:
            return "x"
    except ValueError:
        pass

    if normalize_confidence_interval_prediction(prediction) != "x":
        return "x"
    return prediction


def parse_model_output(text):
    header = "[table] "
    if not text.startswith(header):
        return []
    text = text[len(header):]

    bottom_index = text.rfind(";")
    if bottom_index == -1:
        return []
    text = text[:bottom_index]

    phrase_list = text.split(";")
    table = []
    header_list = ["variant", "disease", "odds ratio", "confidence interval", "p-value"]

    for phrase in phrase_list:
        phrase = phrase.strip()
        row = []
        empty_row = True

        for header in header_list:
            li = phrase.find(header)
            if li == -1:
                cell = "x"
            else:
                phrase = phrase[li+len(header):]
                ri = phrase.find(",")
                if ri == -1:
                    ri = len(phrase)
                cell = phrase[:ri].strip()
                phrase = phrase[ri + 1:]
                if cell:
                    empty_row = False
                else:
                    cell = "x"

            if header == "odds ratio":
                cell = normalize_odds_ratio_prediction(cell)
            elif header == "confidence interval":
                cell = normalize_confidence_interval_prediction(cell)
            elif header == "p-value":
                cell = normalize_p_value_prediction(cell)

            row.append(cell)

        if not empty_row:
            table.append(row)

    return table


def collect_batch_result(
        source_file, all_source_dir, all_target_dir, target_dir,
        complete_triplet_only, indent, variant_tag, disease_tag,
):
    ps_to_table = {}
    batch_list = sorted(os.listdir(all_target_dir), key=lambda n: int(n.split("_")[0]))

    for batch_name in batch_list:
        batch_source_file = os.path.join(all_source_dir, f"{batch_name}.csv")
        batch_target_file = os.path.join(all_target_dir, batch_name, "target.txt")
        input_list = read_csv(batch_source_file, "csv", write_log=False)[1:]
        output_list = read_lines(batch_target_file, write_log=False)
        assert len(input_list) == len(output_list)

        for model_input, model_output in zip(input_list, output_list):
            pmid, sent_id, _, _ = model_input
            if pmid == "x":
                continue
            sent_id = int(sent_id)
            table = parse_model_output(model_output)
            ps_to_table[(pmid, sent_id)] = table

    data = read_json(source_file)
    tuples = 0
    complete_triplets = 0
    sentences = 0

    for datum in data:
        pmid = datum["pmid"]
        sent_id = datum["sent_id"]
        table = ps_to_table.get((pmid, sent_id), [])

        if table:
            complete_triplet_table = [
                row
                for row in table
                if "x" not in row[:3]
            ]
            if complete_triplet_only:
                table = complete_triplet_table
            tuples += len(table)
            complete_triplets += len(complete_triplet_table)
            if table:
                sentences += 1

        # post-hoc matching of model output and mention name by ignoring spaces
        type_to_name_set = {variant_tag: set(), disease_tag: set()}
        type_key_name = {variant_tag: {}, disease_tag: {}}
        for mention in datum["mention_list"]:
            _type = mention["type"]
            if _type not in type_key_name:
                continue
            name = mention["name"]
            key = name.replace(" ", "")
            type_to_name_set[_type].add(name)
            type_key_name[_type][key] = name
        for row in table:
            variant = row[0]
            disease = row[1]
            if variant not in type_to_name_set[variant_tag]:
                v_key = variant.replace(" ", "")
                row[0] = type_key_name[variant_tag].get(v_key, variant)
            if disease not in type_to_name_set[disease_tag]:
                d_key = disease.replace(" ", "")
                row[1] = type_key_name[disease_tag].get(d_key, disease)

        datum["var_dis_or_ci_pv"] = table

    logger.info(f"{sentences:,} sentences with triplets")
    logger.info(f"{tuples:,} tuples")
    logger.info(f"{complete_triplets:,} complete triplets")

    target_file = os.path.join(target_dir, "target.json")
    write_json(target_file, data, indent=indent)
    return


def run_odds_ratio_extraction(arg):
    os.makedirs(arg.target_dir, exist_ok=True)
    model_input_file = os.path.join(arg.target_dir, "model_input.csv")
    all_source_dir = os.path.join(arg.target_dir, "all_source")
    all_target_dir = os.path.join(arg.target_dir, "all_target")
    indent = arg.indent if arg.indent >= 0 else None

    create_model_input(arg.source_file, model_input_file, arg.complete_triplet_only, arg.variant_tag, arg.disease_tag)
    sentences = create_batch_data(model_input_file, all_source_dir, all_target_dir, arg.batch_size)

    for si in range(0, sentences, arg.batch_size):
        start, end = si, min(si + arg.batch_size, sentences)
        logger.info(f"Running [{start}, {end}]")
        batch_source_file = os.path.join(all_source_dir, f"{start}_{end}.csv")
        batch_target_dir = os.path.join(all_target_dir, f"{start}_{end}")
        run_model(batch_source_file, arg.model_dir, batch_target_dir)

    collect_batch_result(
        arg.source_file, all_source_dir, all_target_dir, arg.target_dir,
        arg.complete_triplet_only, indent, arg.variant_tag, arg.disease_tag,
    )
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="./source.json")
    parser.add_argument("--target_dir", type=str, default="./target_dir")
    parser.add_argument("--model_dir", type=str, default="./model")

    parser.add_argument("--variant_tag", type=str, default="mutation")
    parser.add_argument("--disease_tag", type=str, default="disease")
    parser.add_argument("--complete_triplet_only", action="store_true")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--indent", type=int, default=2)

    arg = parser.parse_args()
    run_odds_ratio_extraction(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
