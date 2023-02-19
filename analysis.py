##
from utils.dataset_utils import get_result_txt, preprocess_txt, preprocess2sentence, get_dataset
from datasets import load_dataset
dtype = "wikitext"
filename = "watermarked.txt"

list_of_files = [f'results/ours/{dtype}/new/dep/{filename}', f'results/ours/{dtype}/new/dep-wo-cc/{filename}',
                 f'results/context-ls/{dtype}/paper/{filename}']
list_of_watermarks = [get_result_txt(i) for i in list_of_files]
list_of_sets = []

corpus, test_corpus, num_sample = get_dataset(dtype)
test_cover_texts = preprocess_txt(test_corpus)
test_cover_texts = preprocess2sentence(test_cover_texts, dtype + "-test", 0)

##
from spacy import displacy
from pathlib import Path
from models.kwd import KeywordExtractor

sample_sentence = test_cover_texts[1][9]
visual_options = {'compact': True, "distance": 100, "word_spacing": 40}
svg = displacy.render(sample_sentence, style="dep", options=visual_options)
output_path = Path("./visualization/fig/dep-tree.svg")
with output_path.open("w", encoding="utf-8") as fh:
    fh.write(svg)


keyword_module = KeywordExtractor(ratio=0.3)
all_keywords, entity_keywords = keyword_module.extract_keyword([sample_sentence])

##
for idx, watermarked in enumerate(list_of_watermarks):
    embedded_indices = []
    lucky_cnt = 0
    embedded_cnt = 0
    for idx, (c_idx, sen_idx, sub_idset, sub_idx, clean_wm_text, key, msg) in enumerate(watermarked):
        if len(msg) > 0:
            embedded_cnt += 1
            original = test_cover_texts[c_idx][sen_idx].text
            compared = clean_wm_text
            if original.replace(" ", "").strip() == compared.replace(" ", "").strip():
                lucky_cnt += 1
            hash_str = f"{c_idx},{sen_idx}"
            hash_str = f"{idx}"
            embedded_indices.append(hash_str)

    print(lucky_cnt / embedded_cnt)
    print(embedded_cnt / len(watermarked))
    print("\n")
    embedded_indices = set(embedded_indices)
    list_of_sets.append(embedded_indices)

##
common = list_of_sets[0]
for idx, _ in enumerate(list_of_sets[:-1]):
    common = common.intersection(list_of_sets[idx+1])

print(f"{len(common)} watermark samples available")

NAME = "samples"
SAVE_DIR = f"samples/{dtype}-{NAME}.csv"
f = open(SAVE_DIR, "w", newline="")

import csv
DELIMITER = '|'
wr = csv.writer(f, delimiter=DELIMITER, quoting=csv.QUOTE_MINIMAL)
result = []

result.append(["idx", "Original"] + list_of_files)
watermarked = list_of_watermarks[0]
for idx, (c_idx, sen_idx, sub_idset, sub_idx, clean_wm_text, key, msg) in enumerate(watermarked):
    if str(idx) in common:
        line = []
        print(idx)
        line.append(str(idx))
        print("Original")
        print(test_cover_texts[c_idx][sen_idx], end="\n\n")
        line.append(test_cover_texts[c_idx][sen_idx].text)
        print(list_of_files[0])
        print(clean_wm_text, end="\n\n")
        line.append(clean_wm_text)
        for f_idx, wm_others in enumerate(list_of_watermarks[1:]):
            c_idx, sen_idx, sub_idset, sub_idx, clean_wm_text, key, msg = wm_others[idx]
            print(list_of_files[f_idx+1])
            print(clean_wm_text, end="\n\n")
            line.append(clean_wm_text)
        result.append(line)
        # wr.writerow(line)
        print("\n")

import random
random.seed(1230)
random.shuffle(result)
wr.writerow(["idx", "Original", "dep", "wo-cc", "context-ls"])
wr.writerows(result)
f.close()

##
dtype = "wuthering_heights"
filename = "watermarked.txt"
NAME = "samples"
SAVE_DIR = f"./samples/{dtype}-{NAME}.csv"
result = []
with open(SAVE_DIR, "r") as f:
    result = f.readlines()

result_parsed = [row.split("|") for row in result]

# for row in result_parsed[1:]:
#     original = row[1]
#     diff_marked = ""
#     for method in row[2]:
#         pass
#
# cnt = 0
# for row in result_parsed[1:]:
#     original = row[1]
#     compared = row[4]
#     if original.replace(" ", "").strip() == compared.replace(" ", "").strip():
#         cnt += 1

# print(cnt / (len(result_parsed)-1))

##
import difflib
sample_idx = 14
original = result_parsed[sample_idx][1]
print(original)
for i in range(2,5):
    compared = result_parsed[sample_idx][i]
    print(result_parsed[0][i])
    for i,s in enumerate(difflib.ndiff(compared, original)):
        if s[0] == "-" or s[0] == "+":
            print(i, s)
    print("\n")

##
import pandas as pd
import random
dtype = "wuthering_heights"
df = pd.read_csv(f"./samples/{dtype}-annotated.csv")
new_df = []

random.seed(0)
for idx in range(len(df)):
    row = df.iloc[idx].to_list()[1:]
    shuffled_indices = list(range(len(row)))
    random.shuffle(shuffled_indices)
    indices_str = " ".join([str(x) for x in shuffled_indices])
    shuffled_row = [row[idx].replace("*", "") for idx in shuffled_indices]
    shuffled_row.append(indices_str)
    new_df.append(shuffled_row)

column_names = [f"method{i}" for i in range(len(row))]
new_df = pd.DataFrame(data=new_df, columns=column_names+['order'])
new_df.to_csv(f"./samples/amt/{dtype}-part1.csv")

new_df = []

for idx in range(len(df)):
    row = df.iloc[idx].to_list()[1:]
    original = row[0]
    others = row[1:]
    shuffled_indices = list(range(len(others)))
    random.shuffle(shuffled_indices)
    indices_str = " ".join([str(x) for x in shuffled_indices])
    shuffled_row = [others[idx] for idx in shuffled_indices]
    shuffled_row.append(indices_str)
    original = "Reference: " + original
    shuffled_row.insert(0, original)
    new_df.append(shuffled_row)

column_names = [f"method{i}" for i in range(len(row))]
new_df = pd.DataFrame(data=new_df, columns=column_names+['order'])
new_df.to_csv(f"./samples/amt/{dtype}-part2.csv")

##
# preprocess and analyze AMT results
import pandas as pd
import numpy as np

# preprocess data
original_df = pd.read_csv("./samples/amt/cw-results.csv")
df = pd.read_csv("./samples/amt/cw-results.csv")
values = df.iloc[:, 3:].values
for ridx in range(values.shape[0]):
    for cidx in range(values.shape[1]):
        if not isinstance(values[ridx, cidx], int):
            try:
                v = int(values[ridx,cidx])
            except:
                all_ans = list(map(int, values[ridx,cidx].split(";")))
                v = sum(all_ans) / len(all_ans)
            values[ridx, cidx] = v

df = pd.DataFrame(values, columns=df.columns[3:]) # Part 1: 5*30 ; Part 2: 4*30


# get accuracy on the dummy questions
part2_df = pd.read_csv("./samples/amt/part2-agg.csv")
original = part2_df['method0'].apply(lambda x: x.replace("Reference: ", ""))
others = part2_df.iloc[:, range(2, 6)]

# find indices of the dummy questions (Part 2, those that are identical with reference)
dummy_q_idx = []
flattened_idx = 0
for r_idx, row in enumerate(others.values):
    for sample in row:
        if sample.replace("*", "").strip() == original[r_idx].strip():
            dummy_q_idx.append(flattened_idx)
        flattened_idx += 1

# compute accuracy for each row
part2_offset = 5*30   # start of Part2
dummy_q_idx = [x+part2_offset for x in dummy_q_idx]
acc = ((df.iloc[:, dummy_q_idx] == 5)|(df.iloc[:, dummy_q_idx] == 4)).sum(axis=1) / len(dummy_q_idx)
print(acc)
acc_threshold = 0.5
df = df[acc > acc_threshold]  # filter out those with acc with lower than 0.90
print(df.shape[0])
# df = df.iloc[[7],:]
# Summarize score by methods
def restore_order(answers, ordering):
    buffer = {}
    num_col = answers.shape[-1]
    replaced = [False for _ in range(num_col)]
    for c_idx in range(num_col):
        tmp = answers[:, ordering[c_idx]].copy()
        buffer[ordering[c_idx]] = tmp
        replaced[ordering[c_idx]] = True
        if replaced[c_idx]:
            d = buffer[c_idx]
        else:
            d = answers[:, c_idx]
        answers[:, ordering[c_idx]] = d
    return answers

part1_ordering = pd.read_csv("./samples/amt/part1-agg.csv")['order'].values
part1_result = []
for q_idx in range(0, 30):
    start, end = 5*q_idx, 5*(q_idx+1)
    # print(df.iloc[:, range(start, end)].columns)
    answers = df.iloc[:, range(start, end)].values
    ordering = list(map(int, part1_ordering[q_idx].split()))
    # re-ordered to (original, dep, dep-wo-cc, context-ls, awt)
    answers = restore_order(answers, ordering)
    mean = answers.mean(0)
    part1_result.append(mean[:, np.newaxis])

# 2d-array of (num. method, num. question)
part1_result = np.concatenate(part1_result, axis=1)
part1_result = (part1_result - part1_result[[0],:]).astype(float)

try:
    part1_summary = {'mean': part1_result.mean(1), 'std': part1_result.std(1)}
except:
    part1_summary = {'mean': part1_result.mean(1), 'std': None}

print(part1_summary)


part2_ordering = pd.read_csv("./samples/amt/part2-agg.csv")['order'].values
part2_result = []
for q_idx in range(0, 30):
    start, end = 4 * q_idx + part2_offset, 4 * (q_idx + 1) + part2_offset
    answers = df.iloc[:, range(start, end)].values
    ordering = list(map(int, part2_ordering[q_idx].split()))
    # re-ordered to (original, dep, dep-wo-cc, context-ls, awt)
    answers = restore_order(answers, ordering)
    mean = answers.mean(0)
    part2_result.append(mean[:, np.newaxis])

part2_result = np.concatenate(part2_result, axis=1).astype(float)
try:
    part2_summary = {'mean': part2_result.mean(1), 'std': part2_result.std(1)}
except:
    part2_summary = {'mean': part2_result.mean(1), 'std': None}
print(part2_summary)

##
import os.path
from itertools import product
from datasets import load_dataset
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import random
import re
import spacy
import string
import sys

import torch
from sentence_transformers import SentenceTransformer, util

from models.watermark import InfillModel
from utils.dataset_utils import preprocess2sentence, preprocess_txt
from utils.logging import getLogger
from utils.metric import Metric
from config import WatermarkArgs, GenericArgs, stop

from collections import defaultdict
import pickle
random.seed(1230)

if False:
    infill_parser = InfillArgs()
    generic_parser = GenericArgs()
    infill_args, _ = infill_parser.parse_known_args()
    generic_args, _ = generic_parser.parse_known_args()
    model = InfillModel(infill_args)
    DEBUG_MODE = infill_args.debug_mode


    dtype = generic_args.dtype
    start_sample_idx = 0
    num_sample = generic_args.num_sample
    # corpus = load_dataset(dtype)['train']['text']
    # cover_texts = [t.replace("\n", " ") for t in corpus]
    # cover_texts = preprocess_imdb(cover_texts)
    # cover_texts = preprocess2sentence(cover_texts, dtype + "-train", start_sample_idx, num_sample,
    #                                   spacy_model=infill_args.spacy_model)
    cover_texts, _ = model.return_dataset()

    result = defaultdict(list)
    examples = defaultdict(list)
    spacy_tokenizer = spacy.load("en_core_web_sm")

    for c_idx, sentences in enumerate(cover_texts[:500]):
        for s_idx, sen in enumerate(sentences):
            sen = spacy_tokenizer(sen.text)
            structures = [t.pos_ for t in sen]
            tokens = [t for t in sen]
            for m_idx, struc in enumerate(structures):
                # if str != "dep": # "dep" means unknown dependency
                agg_cwi, agg_probs, mask_idx_pt, inputs = model.fill_mask(sen, [m_idx], train_flag=False, embed_flag=False)
                candidate_texts, candidate_text_jp = model.generate_candidate_sentence(agg_cwi, agg_probs, mask_idx_pt, inputs)
                entail_score = None
                if candidate_texts:
                    entail_score, _ = model.compute_nli(candidate_texts, candidate_text_jp, sen.text)
                if entail_score is not None:
                    result[struc].extend(entail_score.tolist())
                    examples[struc].append([sen.text, [t for t in sen][m_idx].text,
                                                        candidate_texts,
                                                        entail_score.tolist()])


    as_lists = []
    for k, v in result.items():
        mean_score = sum(v) / len(v)
        num_sample = len(v)
        as_lists.append([k, f"{mean_score:.3f}", f"{num_sample}"])

    as_lists.sort(key=lambda x: x[1], reverse=True)
    for struc, mean, num_sample in as_lists:
        print(f"{struc} {mean} for {num_sample} samples")


    savedir = "analysis/pos_result.txt"
    with open(savedir, "w") as wr:
        for line in as_lists:
            wr.write("\t".join(line) + "\n")

    savedir = "analysis/pos_examples.pkl"
    with open(savedir, "wb") as wr:
        pickle.dump(examples, wr)

else:
    with open("./analysis/dep_result.txt", "r") as f:
        result = f.readlines()


    deps = []
    for r in result:
        d = r.split("\t")[0]
        print(r"'"+d+r"',", end=' ')

