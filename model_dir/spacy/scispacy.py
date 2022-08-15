import re
import csv
import sys
import json
import logging
import argparse
from collections import defaultdict
import spacy
from spacy.symbols import DET,PROPN,NOUN,PRON
from spacy.errors import Errors
from spacy.tokens import Doc, Span, Token
from typing import Union, Iterator, Tuple
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


def get_masked_sentence_list(data, start, end, max_sentence_length=3000, max_tokens=128, max_token_length=500):
    sentence_list = []
    di_list = []
    detokenizer = TreebankWordDetokenizer()

    for di in range(start, end):
        datum = data[di]
        mention_list = datum["mention_list"]

        if len(mention_list) < 2:
            continue

        elif "real_pos" in mention_list[0]:
            masked_sentence = [c for c in datum["sentence"]]
            for mi, mention in enumerate(mention_list):
                ci, cj = mention["real_pos"]
                if ci == -1:
                    continue
                masked_sentence[ci] = f"ENT{mi}ITY"
                for i in range(ci + 1, cj):
                    masked_sentence[i] = None
            masked_sentence = [
                c
                for c in masked_sentence[:max_sentence_length]
                if c
            ]
            masked_sentence = "".join(masked_sentence)

        else:
            masked_token_list = [token for token in datum["token_list"]]
            for mi, mention in enumerate(datum["mention_list"]):
                ti, tj = mention["pos"]
                masked_token_list[ti] = f"ENT{mi}ITY"
                for i in range(ti + 1, tj):
                    masked_token_list[i] = None
            masked_token_list = [
                token[:max_token_length]
                for token in masked_token_list[:max_tokens]
                if token
            ]
            masked_sentence = detokenizer.detokenize(masked_token_list)

        sentence_list.append(masked_sentence)
        di_list.append(di)

    return sentence_list, di_list


def add_scispacy_data(data, start, end):
    logger.info(f"scispacy_ore_tool: parse [{start:,}, {end:,}]")

    # mask entity mentions
    sentence_list, di_list = get_masked_sentence_list(data, start, end)

    nlp = spacy.load(
        "en_core_sci_sm",
        exclude=["lemmatizer", "ner"],
    )

    # make sure entities are tagged as proper noun
    ruler = nlp.get_pipe("attribute_ruler")
    patterns = [[{
        "TEXT": {
            "REGEX": r"(ENT\d+ITY)"
        }
    }]]
    attrs = {"TAG": "NNP", "POS": "PROPN"}
    ruler.add(patterns=patterns, attrs=attrs, index=0)

    # add scispacy annotation to data
    offset = 0
    for doc in nlp.pipe(sentence_list):
        di = di_list[offset]
        data[di]["doc"] = doc
        offset += 1
    return


def match_mention(h, r, t, mention_list, mention_expression):
    h_match_list = mention_expression.findall(h)
    r_match_list = mention_expression.findall(r)
    t_match_list = mention_expression.findall(t)

    # Must be exactly one ENTITY in head, zero in relation, one in tail
    if len(h_match_list) != 1 or len(r_match_list) != 0 or len(t_match_list) != 1:
        return None
    h_match = h_match_list[0]
    t_match = t_match_list[0]
    # head entity must not equal tail entity
    if h_match == t_match:
        return None

    perfect_match = h_match == h and t_match == t
    h_mi = int(h_match[3:-3])
    t_mi = int(t_match[3:-3])

    # undo ENTITY masking
    h_name = mention_list[h_mi]["name"]
    t_name = mention_list[t_mi]["name"]
    h = mention_expression.sub(lambda _: h_name, h)
    t = mention_expression.sub(lambda _: t_name, t)

    return h_mi, t_mi, h, t, perfect_match


def get_root(token):
    # follow conjunction and apposition links
    while True:
        if token.dep_ in ["conj", "appos"]:
            token = token.head
        else:
            break
    return token


def get_chunk_text(chunk, junk_token_set):
    tokens = len(chunk)

    for ti in range(tokens):
        if chunk[ti].text not in junk_token_set:
            break
    else:
        return ""

    for tj in range(tokens - 1, -1, -1):
        if chunk[tj].text not in junk_token_set:
            break
    else:
        return ""

    return chunk[ti:tj + 1].text


def get_negation(token):
    neg_list = []
    for child in token.lefts:
        if child.dep_ == "neg":
            neg = str(child)
            if neg == "n't":
                neg = "not"
            neg_list.append(neg)
    neg_prefix = " ".join(neg_list)
    return neg_prefix


## added to reflect scispacy preposition (PART, ADP) pos tag and dependency 
def get_prep(token):
    token=get_root(token)
    if token.left_edge.dep_ == "case" and (token.left_edge.pos_ == "ADP" or token.left_edge.pos_ == "PART"): 
        return token.left_edge
    return ""   

## add to get customized noun chunks for scispacy
def noun_chunks(doclike: Union[Doc, Span]) -> Iterator[Tuple[int, int, int]]:
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    """
    labels = [
        "oprd",
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
        "nmod",
    ]
    doc = doclike.doc  # Ensure works on both Doc and Span.
    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    np_label = doc.vocab.strings.add("NP")
    prev_end = -1
    for i, word in enumerate(doclike):
        if word.pos not in (NOUN, PROPN, PRON, DET):
            continue
        # Prevent nested chunks from being produced
        if word.left_edge.i <= prev_end:
            continue
        if word.dep in np_deps:
            prev_end = word.i
            if word.dep_=="nmod" and word.left_edge.dep_ == "case":
                yield word.left_edge.i+1, word.i + 1, np_label
            else:
                yield word.left_edge.i, word.i + 1, np_label 
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                prev_end = word.i
                yield word.left_edge.i, word.i + 1, np_label

SYNTAX_ITERATORS = {"noun_chunks": noun_chunks}
Doc.set_extension("custom_noun_chunks", getter=noun_chunks,force=True)


def add_relation_data(data, start, end):
    logger.info("scispacy: add_relation_data()")

    mention_expression = re.compile(r"ENT\d+ITY")
    junk_token_set = {
        ",", ".", ":", "?", "!", ";",
        "'", '"', "‘", "’", "“", "”",
        "(", ")", "[", "]", "{", "}",
        "i.e.",
    }
    sentences_with_triplets = 0
    triplets = 0
    perfect_triplets = 0

    for di in range(start, end):
        datum = data[di]
        if "doc" not in datum:
            datum["triplet_list"] = []
            continue
        doc = datum["doc"]
        del datum["doc"]
        possible_relation = defaultdict(lambda: defaultdict(lambda: []))

        new_chunk_list = []

        for start,end,label in doc._.custom_noun_chunks:
            new_chunk_list.append(doc[start:end])

        root_to_noun_chunk = {
            chunk.root: chunk
            for chunk in new_chunk_list
        }

        for chunk in new_chunk_list:
            root = get_root(chunk.root)
            if mention_expression.search(chunk.text) is None:
                continue
            ## when the sentence itself is the whole noun chunk
            if root.head == root:
                continue
            possible_relation[root.head][root.dep_].append({"chunk": chunk})
            head = chunk.root.head
            if head not in root_to_noun_chunk:  # head_chunk
                continue
            head_chunk = root_to_noun_chunk[head]
            ## cant be head_chunk or chunk because that's a span, left_edge only works on token
            leftmost_child = root.left_edge
            if leftmost_child.dep_ != "case" or leftmost_child.text != "of":
                continue
            if root.dep_!="nmod":
                continue
            full_chunk = doc[head_chunk.start:chunk.end]
            del possible_relation[root.head]
            root = get_root(head_chunk.root)
            if root.head == root:
                continue
            possible_relation[root.head][root.dep_].append({"chunk": full_chunk})
        
        mention_list = datum["mention_list"]
        triplet_list = []

        for relation in possible_relation:
            ## active (object's dependency: dobj)
            if "nsubj" in possible_relation[relation] and "dobj" in possible_relation[relation]:
                for subj in possible_relation[relation]["nsubj"]:
                    for obj in possible_relation[relation]["dobj"]:
                        h = get_chunk_text(subj["chunk"], junk_token_set)
                        r = str(relation)
                        t = get_chunk_text(obj["chunk"], junk_token_set)
                        n = get_negation(relation)
                        p = get_prep(obj["chunk"].root)
                        if n and p: 
                            r = f"{n} {r} {p}"
                        if n and not p:
                            r = f"{n} {r}"
                        if p and not n: 
                            r = f"{r} {p}"

                        match = match_mention(h, r, t, mention_list, mention_expression)
                        if match is None:
                            continue
                        h_mi, t_mi, h, t, perfect_match = match
                        triplet_list.append({
                            "h_mention": h_mi,
                            "t_mention": t_mi,
                            "triplet": (h, r, t),
                            "perfect_match": perfect_match
                        })
            ## active (object's dependency: nmod) 
            if "nsubj" in possible_relation[relation] and "nmod" in possible_relation[relation]:
                for subj in possible_relation[relation]["nsubj"]:
                    for obj in possible_relation[relation]["nmod"]:
                        h = get_chunk_text(subj["chunk"], junk_token_set)
                        r = str(relation)
                        t = get_chunk_text(obj["chunk"], junk_token_set)
                        n = get_negation(relation)
                        p = get_prep(obj["chunk"].root)
                        if n and p: 
                            r = f"{n} {r} {p}"
                        if n and not p:
                            r = f"{n} {r}"
                        if p and not n: 
                            r = f"{r} {p}"        

                        match = match_mention(h, r, t, mention_list, mention_expression)
                        if match is None:
                            continue
                        h_mi, t_mi, h, t, perfect_match = match
                        triplet_list.append({
                            "h_mention": h_mi,
                            "t_mention": t_mi,
                            "triplet": (h, r, t),
                            "perfect_match": perfect_match
                        })
            ## passive sentence structure
            if "nsubjpass" in possible_relation[relation] and "nmod" in possible_relation[relation]:
                for subjpass in possible_relation[relation]["nsubjpass"]:
                    for obj in possible_relation[relation]["nmod"]:
                        h = get_chunk_text(subjpass["chunk"], junk_token_set)
                        r = str(relation)
                        t = get_chunk_text(obj["chunk"], junk_token_set)
                        n = get_negation(relation)
                        p = get_prep(obj["chunk"].root)
                        if n and p: 
                            r = f"{n} {r} {p}"
                        ## passive sentence has preposition
                        if n and not p:
                            continue
                        if p and not n: 
                            r = f"{r} {p}"
                        ## passive sentence has preposition
                        if not p:
                            continue

                        match = match_mention(h, r, t, mention_list, mention_expression)
                        if match is None:
                            continue
                        h_mi, t_mi, h, t, perfect_match = match
                        triplet_list.append({
                            "h_mention": h_mi,
                            "t_mention": t_mi,
                            "triplet": (h, r, t),
                            "perfect_match": perfect_match
                        })
                       
        datum["triplet_list"] = triplet_list
        if triplet_list:
            sentences_with_triplets += 1
            triplets += len(triplet_list)
            for triplet in triplet_list:
                if triplet["perfect_match"]:
                    perfect_triplets += 1

    return sentences_with_triplets, triplets, perfect_triplets


def run_scispacy_relation_extraction(arg):
    data = read_json(arg.source_file)
    sentences = len(data)

    if not arg.use_cpu:
        spacy.require_gpu()

    sentences_with_triplets = 0
    triplets = 0
    perfect_triplets = 0


    for start in range(0, sentences, arg.batch_size):
        end = min(start + arg.batch_size, sentences)
        add_scispacy_data(data, start, end)
        ret = add_relation_data(data, start, end)
        sentences_with_triplets += ret[0]
        triplets += ret[1]
        perfect_triplets += ret[2]

    logger.info(f"scispacy_ore_tool: {sentences_with_triplets:,} sentences with triplets")
    logger.info(f"scispacy_ore_tool: {triplets:,} triplets")
    logger.info(f"scispacy_ore_tool: {perfect_triplets:,} perfect triplets")

    indent = arg.indent if arg.indent >= 0 else None
    write_json(arg.target_file, data, indent=indent)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="./source.json")
    parser.add_argument("--target_file", type=str, default="./target.json")
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--use_cpu", action="store_true")
    arg = parser.parse_args()
    run_scispacy_relation_extraction(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
