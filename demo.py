import os.path
from functools import reduce
from itertools import product
import math
import re
import spacy
import string

import torch

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
from utils.logging import getLogger
from utils.metric import Metric

infill_parser = WatermarkArgs()
infill_parser.add_argument("--custom_keywords", type=str)
generic_parser = GenericArgs()
infill_args, _ = infill_parser.parse_known_args()
infill_args.mask_select_method = "grammar"
infill_args.mask_order_by = "dep"
infill_args.exclude_cc = True
generic_args, _ = generic_parser.parse_known_args()

from utils.dataset_utils import preprocess2sentence
raw_text = """
Recent years have witnessed a proliferation of valuable original natural language contents found in subscription-based 
media outlets, web novel platforms, and outputs of large language models. However, these contents are susceptible to 
illegal piracy and potential misuse without proper security measures. This calls for a secure watermarking system to 
guarantee copyright protection through leakage tracing or ownership identification. 
To effectively combat piracy and protect copyrights, a multi-bit watermarking framework should be able to embed 
adequate bits of information and extract the watermarks in a robust manner despite possible corruption. 
In this work, we explore ways to advance both payload and robustness by following a well-known proposition from image 
watermarking and identify features in natural language that are invariant to minor corruption. Through a systematic 
analysis of the possible sources of errors, we further propose a corruption-resistant infill model. 
Our full method improves upon the previous work on robustness by +16.8% point on average on four datasets, 
three corruption types, and two corruption ratios
"""
cover_texts = preprocess2sentence([raw_text], corpus_name="custom",
                                  start_sample_idx=0, cutoff_q=(0.0, 1.0), use_cache=False)
# you can add your own entity / keyword that should NOT be masked.
# This list will be need to be saved when extracting
infill_args.custom_keywords = ["watermarking", "watermark"]

DEBUG_MODE = generic_args.debug_mode
dtype = "custom"
dirname = f"./results/ours/{dtype}/{generic_args.exp_name}"
start_sample_idx = 0
num_sample = generic_args.num_sample

spacy_tokenizer = spacy.load(generic_args.spacy_model)
if "trf" in generic_args.spacy_model:
    spacy.require_gpu()
infill_args.dtype = None
model = InfillModel(infill_args, dirname=dirname)

bit_count = 0
word_count = 0
upper_bound = 0
candidate_kwd_cnt = 0
kwd_match_cnt = 0
mask_match_cnt = 0
sample_cnt = 0
one_cnt = 0
zero_cnt = 0

device = torch.device("cuda")
metric = Metric(device, **vars(generic_args))

if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)

logger = getLogger("DEMO",
                   dir_=dirname,
                   debug_mode=DEBUG_MODE)


result_dir = os.path.join(dirname, "watermarked.txt")
if not os.path.exists(result_dir):
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)


for c_idx, sentences in enumerate(cover_texts):
    corpus_level_watermarks = []
    for s_idx, sen in enumerate(sentences):
        sen = spacy_tokenizer(sen.text.strip())
        all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
        keyword = all_keywords[0]
        ent_keyword = entity_keywords[0]

        agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                              train_flag=False, embed_flag=True)
        # check if keyword & mask_indices matches
        valid_watermarks = []
        candidate_kwd_cnt = 0
        tokenized_text = [token.text_with_ws for token in sen]

        if len(agg_cwi) > 0:
            for cwi in product(*agg_cwi):
                wm_text = tokenized_text.copy()
                for m_idx, c_id in zip(mask_idx, cwi):
                    wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx])

                wm_tokenized = spacy_tokenizer("".join(wm_text).strip())

                # extract keyword of watermark
                wm_keywords, wm_ent_keywords = model.keyword_module.extract_keyword([wm_tokenized])
                wm_kwd = wm_keywords[0]
                wm_ent_kwd = wm_ent_keywords[0]
                wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)

                kwd_match_flag = set([x.text for x in wm_kwd]) == set([x.text for x in keyword])
                if kwd_match_flag:
                    kwd_match_cnt += 1

                # checking whether the watermark can be embedded without the assumption of corruption
                mask_match_flag = len(wm_mask) and set(wm_mask_idx) == set(mask_idx)
                if mask_match_flag:
                    text2print = [t.text_with_ws for t in wm_tokenized]
                    for m_idx in mask_idx:
                        text2print[m_idx] = f"\033[92m{text2print[m_idx]}\033[00m"
                    valid_watermarks.append(text2print)
                    mask_match_cnt += 1

                sample_cnt += 1
                candidate_kwd_cnt += 1

        punct_removed = sen.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
        word_count += len([i for i in punct_removed.split(" ") if i not in stop])
        if len(valid_watermarks) > 1:
            bit_count += math.log2(len(valid_watermarks))
            for vw in valid_watermarks:
                print("".join(vw))

        if len(valid_watermarks) == 0:
            valid_watermarks = [sen.text]

        corpus_level_watermarks.append(valid_watermarks)

        if candidate_kwd_cnt > 0:
            upper_bound += math.log2(candidate_kwd_cnt)

    if word_count:
        logger.info(f"Bpw : {bit_count / word_count:.3f}")

    num_options = reduce(lambda x, y: x * y, [len(vw) for vw in corpus_level_watermarks])
    available_bit = math.floor(math.log2(num_options))
    print(num_options)
    print(f"Input message to embed (max bit:{available_bit}):")
    message = input().replace(" ", "")
    # left pad to available bit if given message is short
    if available_bit > 8:
        print(f"Available bit is large: {available_bit} > 8.. "
              f"We recommend using shorter text segments as it may take a while")
    message = "0" * (available_bit - len(message)) + message
    if len(message) > available_bit:
        print(f"Given message longer than capacity. Truncating...: {len(message)}>{available_bit}")
        message = message[:available_bit]
    message_decimal = int(message, 2)
    cnt = 0
    watermarked_text = ""
    available_candidates = product(*corpus_level_watermarks)
    while cnt < message_decimal:
        cnt += 1
        watermarked_text = next(available_candidates)

    print("---- Watermarked text ----")
    for wt in watermarked_text:
        print("".join(wt))