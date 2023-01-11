##
from utils.dataset_utils import get_result_txt, preprocess_txt, preprocess2sentence, get_dataset

dtype = "wikitext"
filename = "watermarked.txt"

list_of_files = [f'results/ours/{dtype}/new/dep/{filename}', f'results/ours/{dtype}/new/dep-wo-cc/{filename}',
                 f'results/context-ls/{dtype}/paper/{filename}']
list_of_watermarks = [get_result_txt(i) for i in list_of_files]
list_of_sets = []

corpus, test_corpus, num_sample = get_dataset(dtype)
test_cover_texts = preprocess_txt(test_corpus)
test_cover_texts = preprocess2sentence(test_cover_texts, dtype + "-test", 0)

for idx, watermarked in enumerate(list_of_watermarks):
    embedded_indices = []
    for idx, (c_idx, sen_idx, sub_idset, sub_idx, clean_wm_text, key, msg) in enumerate(watermarked):
        if len(msg) > 0:
            hash_str = f"{c_idx},{sen_idx}"
            hash_str = f"{idx}"
            embedded_indices.append(hash_str)
    embedded_indices = set(embedded_indices)
    list_of_sets.append(embedded_indices)

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
random.shuffle(result)
wr.writerow(["idx", "Original", "dep", "wo-cc", "context-ls"])
wr.writerows(result)
f.close()



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

