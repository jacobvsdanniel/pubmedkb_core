import os
import re
import sys
import csv
import logging
import argparse
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

csv.register_dialect(
    "yolo", delimiter=",", quoting=csv.QUOTE_ALL, quotechar='"', doublequote=True,
    escapechar=None, lineterminator="\n", skipinitialspace=False, strict=True,
)
csv.register_dialect(
    "rbert",
    delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, doublequote=False,
    escapechar=None, lineterminator="\n", skipinitialspace=False, strict=True,
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


def train(base, dataset, max_steps=None):
    if dataset.startswith("snpphena"):
        log_steps = 100
        train_steps = 1000
        lr_steps = 1000
    elif dataset.startswith("anndisge"):
        log_steps = 100
        train_steps = 1000
        lr_steps = 1000
    elif dataset.startswith("disgevar"):
        log_steps = 500
        train_steps = 5000
        lr_steps = 5000
    elif dataset.startswith("nonanndisge"):
        log_steps = 500
        train_steps = 5000
        lr_steps = 5000
    else:
        assert False
    if max_steps:
        train_steps = max_steps
    subprocess.run(["sh", "script/train.sh", base, dataset, str(log_steps), str(train_steps), str(lr_steps)])
    log_file = f"log/train_{base}-{dataset}_log{log_steps}_train{train_steps}_lr{lr_steps}.txt"
    return log_file


def parse_train_log_file_name(log_file):
    log_file = log_file[log_file.rfind("/")+1:]
    assert log_file.startswith("train_")
    assert log_file.endswith(".txt")
    model, log_steps, train_steps, lr_steps = log_file[6:-4].split("_")
    i = model.rfind("-")
    base, dataset = model[:i], model[i+1:]
    log_steps = int(log_steps[3:])
    train_steps = int(train_steps[5:])
    lr_steps = int(lr_steps[2:])
    # logger.info(f"base={base} dataset={dataset} log_steps={log_steps} train_steps={train_steps} lr_steps={lr_steps}")
    return base, dataset, log_steps, train_steps, lr_steps


def parse_train_log_file(log_file):
    logger.info(f"parse_train_log_file(): {log_file}")
    _, _, log_steps, train_steps, _ = parse_train_log_file_name(log_file)

    with open(log_file, "r", encoding="utf8") as f:
        line_list = f.read().splitlines()

    score_list = []
    for li, line in enumerate(line_list):
        if line[-12:-6] == "acc = ":
            acc = float(line[-6:])
            f1 = float(line_list[li+1][-6:])
            score_list.append((f1, acc))
    final_score = score_list.pop()
    assert score_list[-1] == final_score
    scores = len(score_list)
    assert log_steps * scores == train_steps

    best_si = max(range(scores), key=lambda si: score_list[si])
    best_steps = log_steps * (best_si + 1)
    f1, acc = score_list[best_si]
    logger.info(f"best_steps={best_steps} f1={f1} acc={acc}")
    return best_steps, f1, acc


def extract_sentence(source, target):
    source_file = os.path.join("data", source, "train.tsv")
    target_file = os.path.join("data", target, "auto_sentence.txt")
    logger.info(f"extract_sentence(): source={source_file}")
    logger.info(f"extract_sentence(): target={target_file}")

    row_list = read_csv(source_file, "rbert")
    row_list = [[row[1]] for row in row_list]
    write_csv(target_file, "rbert", row_list)
    return


def predict(model, dataset):
    model_dir = os.path.join("model", model)
    input_file = os.path.join("data", dataset, "auto_sentence.txt")
    output_file = os.path.join("data", dataset, "auto_prediction.txt")
    logger.info(f"predict() model_dir={model_dir}")
    logger.info(f"predict() input_file={input_file}")
    logger.info(f"predict() output_file={output_file}")
    subprocess.run(
        [
            "python", "predict.py",
            "--model_dir", model_dir,
            "--input_file", input_file,
            "--output_file", output_file,
        ],
    )
    return


def create_student_dataset(gold_dataset, student_dataset):
    # training data: gold
    gold_train_file = os.path.join("data", gold_dataset, "train.tsv")
    row_list = read_csv(gold_train_file, "rbert")

    # training data: auto
    auto_prediction_file = os.path.join("data", student_dataset, "auto_prediction.txt")
    auto_sentence_file = os.path.join("data", student_dataset, "auto_sentence.txt")
    with open(auto_prediction_file, "r", encoding="utf8") as f:
        prediction_list = f.read().splitlines()
    with open(auto_sentence_file, "r", encoding="utf8") as f:
        sentence_list = f.read().splitlines()
    assert len(sentence_list) == len(prediction_list)
    for prediction, sentence in zip(prediction_list, sentence_list):
        row_list.append([prediction, sentence])

    # training data: student
    student_train_file = os.path.join("data", student_dataset, "train.tsv")
    write_csv(student_train_file, "rbert", row_list)

    # testing data, label, scorer, answer keys
    src_list = [
        os.path.join("data", gold_dataset, "test.tsv"),
        os.path.join("data", gold_dataset, "label.txt"),
        os.path.join("eval", gold_dataset, "scorer.pl"),
        os.path.join("eval", gold_dataset, "answer_keys.txt"),
    ]
    tgt_list = [
        os.path.join("data", student_dataset, "test.tsv"),
        os.path.join("data", student_dataset, "label.txt"),
        os.path.join("eval", student_dataset, "scorer.pl"),
        os.path.join("eval", student_dataset, "answer_keys.txt"),
    ]
    for src, tgt in zip(src_list, tgt_list):
        subprocess.run(["cp", src, tgt])
    return


def self_training(arg):
    # Initialize teacher: tune base model on gold data
    if arg.retrain_gold_model:
        log_file = train(arg.base, arg.gold)
        best_steps, f1, acc = parse_train_log_file(log_file)
        log_file = train(arg.base, arg.gold, max_steps=best_steps)
        best = parse_train_log_file(log_file)
        assert best == (best_steps, f1, acc)
    teacher_model = f"{arg.base}-{arg.gold}"

    result = []
    for si in range(1, 1 + arg.student_iterations):
        # Use teacher to predict labels for auto data
        student_dataset = f"{arg.auto}{si}"
        os.mkdir(os.path.join("data", student_dataset))
        os.mkdir(os.path.join("eval", student_dataset))
        extract_sentence(arg.auto, student_dataset)
        predict(teacher_model, student_dataset)
        create_student_dataset(arg.gold, student_dataset)

        # Train student
        base = teacher_model if arg.inherit_teacher else arg.base
        log_file = train(base, student_dataset)
        best_steps, f1, acc = parse_train_log_file(log_file)
        log_file = train(base, student_dataset, max_steps=best_steps)
        best = parse_train_log_file(log_file)
        assert best == (best_steps, f1, acc)
        result.append(best)

        # Replace teacher by student
        teacher_model = f"{base}-{student_dataset}"

    for si, (steps, f1, acc) in enumerate(result):
        logger.info(f"student{si+1} steps={steps} f1={f1} acc={acc}")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="biobert")
    parser.add_argument("--gold", type=str, default="anndisge")
    parser.add_argument("--auto", type=str, default="nonanndisge")

    parser.add_argument("--inherit_teacher", action="store_true")
    parser.add_argument("--retrain_gold_model", action="store_true")
    parser.add_argument("--student_iterations", type=int, default=3)

    arg = parser.parse_args()
    self_training(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
