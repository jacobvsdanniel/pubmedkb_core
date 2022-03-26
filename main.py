"""
CUDA 10.1;
Optimized for 1 GPU + 13 CPUs;
Please make sure NEN server has started;
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


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


def get_env(venv_dir):
    env = os.environ.copy()
    pi = env["PATH"].find(":")
    assert pi > 0
    bin_dir = os.path.join(venv_dir, "bin")
    env["PATH"] = bin_dir + env["PATH"][pi:]
    env["VIRTUAL_ENV"] = venv_dir
    interpreter = os.path.join(bin_dir, "python")
    return env, interpreter


class Task:
    def __init__(self, name):
        self.name = name
        self.parent_list = []
        self.process = None
        self.finished = False
        self.source = None
        self.target = None
        self.indent = None
        return

    def run(self):
        logger.info(f"Task: {self.name}")

        venv_dir = os.path.join("venv_dir", self.name)
        venv_dir = os.path.abspath(venv_dir)
        env, interpreter = get_env(venv_dir)

        source_file = os.path.abspath(self.source)
        target_dir = os.path.abspath(os.path.join(self.target, self.name))
        command = self.get_command(interpreter, source_file, target_dir, self.indent)
        logger.info(f"Command: {command}")

        model_dir = os.path.join("model_dir", self.name)
        os.makedirs(target_dir, exist_ok=False)
        self.process = subprocess.Popen(command, env=env, cwd=model_dir)
        return

    def get_command(self, abspath_interpreter, abspath_source_file, abspath_target_dir, indent):
        if self.name == "ner":
            command = [
                abspath_interpreter, "main.py",
                "--source_file", abspath_source_file,
                "--target_dir", abspath_target_dir,
                "--indent", indent,
            ]
        elif self.name == "pop":
            command = [
                abspath_interpreter, "main.py",
                "--source_file", abspath_source_file,
                "--target_dir", abspath_target_dir,
                "--model_dir", "model",
                "--number_measure_file", "number_measure.txt",
                "--species_file", "species.txt",
                "--indent", indent,
            ]
        elif self.name == "or":
            command = [
                abspath_interpreter, "main.py",
                "--source_file", abspath_source_file,
                "--target_dir", abspath_target_dir,
                "--model_dir", "model",
                "--variant_tag", "mutation",
                "--disease_tag", "disease",
                "--complete_triplet_only",
                "--batch_size", "512",
                "--indent", indent,
            ]
        elif self.name == "openie":
            command = [
                abspath_interpreter, "main.py",
                "--source_file", abspath_source_file,
                "--target_dir", abspath_target_dir,
                "--jar_dir", "jar_dir/",
                "--batch_size", "2000",
                "--batch_processes", "4",
                "--process_memory", "6g",
                "--indent", indent,
            ]
        elif self.name == "spacy":
            command = [
                abspath_interpreter, "main.py",
                "--source_file", abspath_source_file,
                "--target_file", os.path.join(abspath_target_dir, "target.json"),
                "--batch_size", "50000",
                "--indent", indent,
            ]
        elif self.name == "nen":
            command = [
                abspath_interpreter, "main.py",
                "--source_file", abspath_source_file,
                "--target_file", os.path.join(abspath_target_dir, "target.json"),
                "--batch_size", "10000",
                "--processes", "20",
                "--port", "8888",
                "--indent", indent,
            ]
        elif self.name == "cre":
            command = [
                abspath_interpreter, "main.py",
                "--source_file", abspath_source_file,
                "--target_dir", abspath_target_dir,
                "--variant_tag", "mutation",
                "--disease_tag", "disease",
                "--indent", indent,
            ]
        else:
            assert False
        return command

    def check_finish(self):
        if self.process is None or self.finished:
            return
        ret = self.process.poll()
        if ret is None:
            return
        assert ret == 0
        self.finished = True
        self.remove_temporary_file()
        logger.info(f"Cleanup: {self.name}")
        return

    def remove_temporary_file(self):
        if self.name == "ner":
            tmp_file_list = ["input.txt", "target.tsv"]
            tmp_dir_list = []
        elif self.name == "pop":
            tmp_file_list = ["model_input.csv", "model_output.txt", "all_results.json", "test_results.json"]
            tmp_dir_list = []
        elif self.name == "or":
            tmp_file_list = ["model_input.csv"]
            tmp_dir_list = ["all_source", "all_target"]
        elif self.name == "openie":
            tmp_file_list = []
            tmp_dir_list = ["all_source", "all_target"]
        elif self.name == "spacy":
            tmp_file_list = []
            tmp_dir_list = []
        elif self.name == "nen":
            tmp_file_list = []
            tmp_dir_list = []
        elif self.name == "cre":
            tmp_file_list = ["meta.csv", "input.txt", "output.txt", "output.txt.npy"]
            tmp_dir_list = []
        else:
            assert False

        for tmp_file in tmp_file_list:
            os.remove(os.path.join(self.target, self.name, tmp_file))
        for tmp_dir in tmp_dir_list:
            shutil.rmtree(os.path.join(self.target, self.name, tmp_dir))
        return


def initialize_all_task(source_file, target_dir, todo_name_list, indent):
    name_to_task = {
        name: Task(name)
        for name in ["ner", "pop", "or", "openie", "spacy", "nen", "cre"]
    }
    todo_name_list = set(todo_name_list.split(","))

    for name, task in name_to_task.items():
        task.finished = name not in todo_name_list
        task.target = target_dir
        task.indent = str(indent)

        if name == "ner":
            task.parent_list = []
            task.source = source_file

        elif name == "pop":
            task.parent_list = [name_to_task[name] for name in ["ner"]]
            task.source = source_file

        elif name == "or":
            task.parent_list = [name_to_task[name] for name in ["pop"]]
            if "ner" in todo_name_list:
                task.source = os.path.join(target_dir, "ner", "target.json")
            else:
                task.source = source_file

        elif name == "openie":
            task.parent_list = [name_to_task[name] for name in ["ner"]]
            if "ner" in todo_name_list:
                task.source = os.path.join(target_dir, "ner", "target.json")
            else:
                task.source = source_file

        elif name == "spacy":
            task.parent_list = [name_to_task[name] for name in ["or"]]
            if "ner" in todo_name_list:
                task.source = os.path.join(target_dir, "ner", "target.json")
            else:
                task.source = source_file

        elif name == "nen":
            task.parent_list = [name_to_task[name] for name in ["ner"]]
            if "ner" in todo_name_list:
                task.source = os.path.join(target_dir, "ner", "target.json")
            else:
                task.source = source_file

        elif name == "cre":
            task.parent_list = [name_to_task[name] for name in ["spacy", "nen"]]
            if "nen" in todo_name_list:
                task.source = os.path.join(target_dir, "nen", "target.json")
            else:
                task.source = source_file

    return name_to_task


def collect_result(source_file, target_dir, todo_name_list, indent):
    source_data = read_json(source_file)
    todo_name_list = todo_name_list.split(",")

    for name in todo_name_list:
        task_dir = os.path.join(target_dir, name)
        task_file = os.path.join(task_dir, "target.json")
        task_data = read_json(task_file)
        assert len(source_data) == len(task_data)

        for di, datum in enumerate(source_data):
            if name == "ner" and "nen" not in todo_name_list:
                datum["mention_list"] = task_data[di]["mention_list"]

            elif name == "pop":
                datum["population"] = task_data[di]["population"]

            elif name == "or":
                datum["odds_ratio"] = task_data[di]["var_dis_or_ci_pv"]

            elif name == "openie":
                datum["openie_ore"] = task_data[di]["triplet_list"]

            elif name == "spacy":
                datum["spacy_ore"] = task_data[di]["triplet_list"]

            elif name == "nen":
                datum["mention_list"] = task_data[di]["mention_list"]

            elif name == "cre":
                datum["rbert_cre"] = task_data[di]["triplet_list"]

        os.remove(task_file)
        os.rmdir(task_dir)

    target_file = os.path.join(target_dir, "target.json")
    indent = indent if indent >= 0 else None
    write_json(target_file, source_data, indent)
    return


def run_pubmedkb_core(arg):
    name_to_task = initialize_all_task(arg.source_file, arg.target_dir, arg.task, arg.indent)
    os.makedirs(arg.target_dir, exist_ok=False)

    while True:
        # Check all task status
        all_finished = True
        for _, task in name_to_task.items():
            task.check_finish()
            if not task.finished:
                all_finished = False
        if all_finished:
            break

        # Run tasks who: not running, not finished, parents are all finished
        for _, task in name_to_task.items():
            if task.process is not None or task.finished:
                continue
            for parent in task.parent_list:
                if not parent.finished:
                    break
            else:
                task.run()

        time.sleep(30)

    collect_result(arg.source_file, arg.target_dir, arg.task, arg.indent)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="source.json")
    parser.add_argument("--target_dir", type=str, default="target_dir")
    parser.add_argument("--task", type=str, default="ner,pop,or,openie,spacy,nen,cre")
    parser.add_argument("--indent", type=int, default=2)

    arg = parser.parse_args()
    run_pubmedkb_core(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
