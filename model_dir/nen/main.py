import sys
import json
import time
import logging
import argparse
import requests
import multiprocessing
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
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


def run_model(mention_name, mention_type, port):
    param = json.dumps({
        "text": mention_name,
        "type": mention_type,
    })
    body_data = {"param": param}

    while True:
        try:
            response = requests.post(f"http://localhost:{port}", data=body_data)
        except requests.ConnectionError:
            logger.info("NEN: connection error")
            time.sleep(1)
            continue
        break

    try:
        result = response.json()
    except json.JSONDecodeError:
        return []

    entity_list = []

    for entity in result["denotations"]:
        id_list = [
            mid
            for mid in entity["id"]
            if isinstance(mid, str) and mid != "CUI-less"
        ]
        if id_list:
            entity_list.append(id_list)

    return entity_list


def run_nen(arg):
    source_data = read_json(arg.source_file)
    total_di = len(source_data)
    type_to_mentions = defaultdict(lambda: 0)
    type_to_ids = defaultdict(lambda: 0)
    type_mapping = {
        "VARIANT": "mutation",
        "Chemical": "drug",
    }

    for start in range(0, total_di, arg.batch_size):
        end = min(start + arg.batch_size, total_di)
        logger.info(f"NEN: running [{start:,}, {end:,})")

        # collect mentions
        mention_list = []
        for di in range(start, end):
            for mention in source_data[di]["mention_list"]:
                mt = mention["type"]
                mt = type_mapping.get(mt, mt.lower())
                if mt not in ["gene", "mutation", "disease", "drug"]:
                    mention["id"] = []
                    continue
                type_to_mentions[mt] += 1
                mention["type"] = mt
                mention_list.append(mention)

        # get mention ids with subprocesses
        model_arg_list = [
            (mention["name"], mention["type"], arg.port)
            for mention in mention_list
        ]
        with multiprocessing.Pool(arg.processes) as pool:
            mid_list = pool.starmap(run_model, model_arg_list)

        for mention, mid in zip(mention_list, mid_list):
            mention["id"] = mid
            if mid:
                type_to_ids[mention["type"]] += 1

        log = f"{end:,}/{total_di:,} sentences"
        for mt in ["gene", "mutation", "disease", "drug"]:
            mentions = type_to_mentions[mt]
            ids = type_to_ids[mt]
            log += f"; {ids:,}/{mentions:,} {mt}s"
        logger.info(log)

    indent = arg.indent if arg.indent >= 0 else None
    write_json(arg.target_file, source_data, indent=indent)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="./source.json")
    parser.add_argument("--target_file", type=str, default="./target.json")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--processes", type=int, default=20)
    parser.add_argument("--port", type=str, default="8888")
    parser.add_argument("--indent", type=int, default=2)
    arg = parser.parse_args()

    run_nen(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
