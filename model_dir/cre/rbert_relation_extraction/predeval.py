import os
import re
import sys
import csv
import logging
import argparse
import subprocess
from collections import defaultdict

logger = logging.getLogger(__name__)

csv.register_dialect(
    "yolo",
    delimiter=",",
    quoting=csv.QUOTE_ALL,
    quotechar='"',
    doublequote=True,
    escapechar=None,
    lineterminator="\n",
    skipinitialspace=False,
    strict=True,
)

csv.register_dialect(
    "rbert",
    delimiter="\t",
    quoting=csv.QUOTE_NONE,
    quotechar=None,
    doublequote=False,
    escapechar=None,
    lineterminator="\n",
    skipinitialspace=False,
    strict=True,
)


def read_csv(file, dialect):
    with open(file, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f, dialect=dialect)
        row_list = [row for row in reader]
    return row_list


def write_csv(file, dialect, row_list):
    with open(file, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, dialect=dialect)
        for row in row_list:
            writer.writerow(row)
    return


def predict_and_evaluate(model_dir, sentence_file, eval_dir, label_map=None):

    # Predict
    output_label_file = os.path.join(eval_dir, "proposed_labels.txt")
    subprocess.run(
        [
            "python", "predict.py",
            "--model_dir", model_dir,
            "--input_file", sentence_file,
            "--output_file", output_label_file,
        ],
    )

    # Transform prediction format
    with open(output_label_file, "r", encoding="utf8") as f:
        label_list = f.read().splitlines()
    if label_map is None:
        label_map = {}
    row_list = [
        [8001+li, label_map.get(label, label)]
        for li, label in enumerate(label_list)
    ]
    output_answer_file = os.path.join(eval_dir, "proposed_answers.txt")
    write_csv(output_answer_file, "rbert", row_list)

    # Evaluate
    result_file = os.path.join(eval_dir, "result.txt")
    completed_process = subprocess.run(
        [
            "perl", f"{eval_dir}/scorer.pl",
            f"{eval_dir}/proposed_answers.txt",
            f"{eval_dir}/answer_keys.txt",
        ],
        stdout=subprocess.PIPE,
    )
    stdout = completed_process.stdout.decode("utf8")
    with open(result_file, "w", encoding="utf8") as f:
        f.write(stdout)
    return


def extract_correct_data(gold_file, prediction_file, correct_file):
    row_list = read_csv(gold_file, "rbert")
    with open(prediction_file, "r", encoding="utf8") as f:
        prediction_list = f.read().splitlines()
    assert len(row_list) == len(prediction_list)

    data = []
    for row, prediction in zip(row_list, prediction_list):
        if row[0] == prediction:
            data.append(row)

    write_csv(correct_file, "rbert", data)
    return


def extract_sentence(dataset):
    logger.info(f"extract_sentence() for {dataset}")
    test_file = os.path.join("data", dataset, "test.tsv")
    row_list = read_csv(test_file, "rbert")
    row_list = [[row[1]] for row in row_list]
    sentence_file = os.path.join("data", dataset, "test_sentence.txt")
    write_csv(sentence_file, "rbert", row_list)
    return


def predict(model_dataset, target_dataset):
    logger.info(f"predict() for model={model_dataset} target={target_dataset}")

    # raw prediction
    model_dir = os.path.join("model", model_dataset)
    input_file = os.path.join("data", target_dataset, "test_sentence.txt")
    output_file = os.path.join("eval", target_dataset, "proposed_answers.txt")
    subprocess.run(
        [
            "python", "predict.py",
            "--model_dir", model_dir,
            "--input_file", input_file,
            "--output_file", output_file,
        ],
    )

    # label mapping
    model_label_file = os.path.join("data", model_dataset[model_dataset.rfind("-")+1:], "label.txt")
    with open(model_label_file, "r", encoding="utf8") as f:
        model_label_list = f.read().splitlines()
    target_label_file = os.path.join("data", target_dataset, "label.txt")
    with open(target_label_file, "r", encoding="utf8") as f:
        target_label_list = f.read().splitlines()
    label_map = {}
    for label in model_label_list:
        if label in target_label_list:
            continue
        if label == "VNEG-D(e1,e2)":
            label_map[label] = "Other"
        else:
            assert False

    # transform prediction
    with open(output_file, "r", encoding="utf8") as f:
        prediction_list = f.read().splitlines()
    row_list = [
        [8001+li, label_map.get(prediction, prediction)]
        for li, prediction in enumerate(prediction_list)
    ]
    write_csv(output_file, "rbert", row_list)
    return


def evaluate(dataset):
    logger.info(f"evaluate() for {dataset}")
    completed_process = subprocess.run(
        [
            "perl",
            os.path.join("eval", dataset, "scorer.pl"),
            os.path.join("eval", dataset, "proposed_answers.txt"),
            os.path.join("eval", dataset, "answer_keys.txt"),
        ],
        stdout=subprocess.PIPE,
    )
    stdout = completed_process.stdout.decode("utf8")
    result_file = os.path.join("eval", dataset, "result.txt")
    with open(result_file, "w", encoding="utf8") as f:
        f.write(stdout)

    stdout = stdout.splitlines()
    for li, line in enumerate(stdout):
        if line.startswith("Accuracy"):
            assert line[-1] == "%"
            acc = float(line[-6:-1]) / 100
        elif line.startswith("Micro-averaged result"):
            line = stdout[li+1]
            assert line[-1] == "%"
            f1 = float(line[-6:-1]) / 100
            break
    logger.info(f"f1={f1:.4f} acc={acc:.4f}")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    extract_sentence(args.target)
    predict(args.source, args.target)
    evaluate(args.target)

    model_dir = os.path.join("model", "snpphena_biobert")

    sentence_file = os.path.join("data", "snpphena", "test_sentence.tsv")
    eval_dir = os.path.join("eval", "snpphena")
    # predict_and_evaluate(model_dir, sentence_file, eval_dir)

    sentence_file = os.path.join("data", "disgevar_to_snpphena", "test_sentence.tsv")
    eval_dir = os.path.join("eval", "disgevar_to_snpphena")
    label_map = {"VNEG-D(e1,e2)": "Other"}
    # predict_and_evaluate(model_dir, sentence_file, eval_dir, label_map=label_map)

    gold_file = os.path.join("data", "disgevar_rsentmask", "train.tsv")
    prediction_file = os.path.join("data", "disgevar_rsentmask", "train_prediction_by_snpphena_model.txt")
    correct_file = os.path.join("data", "disgevar_rsentmask", "train_correct_by_snpphena_model.tsv")
    # extract_correct_data(gold_file, prediction_file, correct_file)
    return


if __name__ == "__main__":
    main()
    sys.exit()
