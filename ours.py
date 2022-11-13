import os.path
from itertools import product
from datasets import load_dataset
import math
import random
import re
import spacy
import string
import sys

import torch

from models.infill import InfillModel
from utils.dataset_utils import preprocess2sentence, preprocess_imdb
from utils.logging import getLogger
from config import InfillArgs, stop

parser = InfillArgs()
args = parser.parse_args()
model = InfillModel(args)

dtype = 'imdb'
start_sample_idx = 0
num_sample = 50
corpus = load_dataset(dtype)['test']['text']
cover_texts = [t.replace("\n", " ") for t in corpus]
cover_texts = preprocess_imdb(cover_texts)
cover_texts = preprocess2sentence(cover_texts, dtype + "-test", start_sample_idx, num_sample,
                                  spacy_model="en_core_web_sm")


##
EMBED = False
spacy_tokenizer = spacy.load("en_core_web_sm")
bit_count = 0
word_count = 0

upper_bound = 0
candidate_kwd_cnt = 0
kwd_match_cnt = 0
mask_match_cnt = 0
sample_cnt = 0

one_cnt = 0 
zero_cnt = 0

if EMBED:
    logger = getLogger("EMBED")
    result_dir = f"results/ours/{dtype}.txt"
    if not os.path.exists(result_dir):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    wr = open(result_dir, "w")
    for c_idx, sentences in enumerate(cover_texts):
        for s_idx, sen in enumerate(sentences):
            sen = spacy_tokenizer(sen.text)
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
            # sen_text = sen.text
            logger.debug(f"{c_idx} {s_idx}")
            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]
    
            agg_cwi, mask_idx, mask_word  = model.run_iter(sen, keyword, ent_keyword, train_flag=False, embed_flag=True)
    
            # check if keyword & mask_indices matches
            valid_watermarks= []
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

                    wm_mask = []
                    wm_mask_idx = []
                    for k in wm_kwd:
                        if k.head not in wm_kwd and k.head not in wm_mask and\
                        k.head not in wm_ent_kwd and not k.head.is_punct:
                            wm_mask.append(k.head)
                            wm_mask_idx.append(k.head.i)

                    kwd_match_flag = set([x.text for x in wm_kwd]) == set([x.text for x in keyword])
                    if kwd_match_flag:
                        kwd_match_cnt += 1
                    # checking whether the watermark can be embedded without the assumption of corruption
                    # for extraction, infill should be done
                    mask_match_flag = set(wm_mask_idx) == set(mask_idx)
                    if mask_match_flag:
                        valid_watermarks.append(wm_tokenized.text)
                        mask_match_cnt += 1
                    sample_cnt += 1
                    candidate_kwd_cnt += 1
    
            punct_removed = sen.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])
            # valid_watermarks = valid_watermarks[:len(valid_watermarks) // 2 * 2]
            if len(valid_watermarks) > 1:
                bit_count += math.log2(len(valid_watermarks))
                random_msg_decimal = random.choice(range(len(valid_watermarks)))
                num_digit = math.ceil(math.log2(len(valid_watermarks)))
                random_msg_binary = format(random_msg_decimal, f"0{num_digit}b")

                wm_text = valid_watermarks[random_msg_decimal]
                watermarked_text = "".join(wm_text) if len(mask_idx) > 0 else ""
                message_str = list(random_msg_binary)
                one_cnt += len([i for i in message_str if i =="1"])
                zero_cnt += len([i for i in message_str if i =="0"])

                message_str = ' '.join(message_str) if len(message_str) else ""
                wr.write(f"{c_idx}\t{s_idx}\t \t \t"
                         f"{''.join(wm_text)}\t \t{message_str}\n")
            else:
                original_text = ''.join(tokenized_text)
                wr.write(f"{c_idx}\t{s_idx}\t \t \t"
                         f"{original_text}\t \t \n")

            if candidate_kwd_cnt > 0:
                upper_bound += math.log2(candidate_kwd_cnt)
    
    wr.close()
    logger.info(f"UB Bpw : {upper_bound / word_count:.3f}")
    logger.info(f"Bpw : {bit_count / word_count:.3f}")
    logger.info(f"mask match rate: {mask_match_cnt / sample_cnt:.3f}")
    logger.info(f"kwd match rate: {kwd_match_cnt / sample_cnt :.3f}")
    logger.info(f"zero/one ratio: {zero_cnt / one_cnt :.3f}")
    
    result_dir = "results/ours/bpw.txt"
    if not os.path.exists(result_dir):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    with open(result_dir, "a") as wr:
        wr.write(f"{dtype}-{num_sample}\t {bit_count / word_count}\n")
      
      
##
from utils.misc import get_result_txt, compute_ber

EXTRACT = True

if EXTRACT:
    corrupted_flag = True
    logger = getLogger(f"EXTRACT-CORRUPTED" if corrupted_flag else "EXTRACT")
    dtype = "imdb"
    start_sample_idx = 0
    num_sample = 50
    result_dir = f"results/ours/{dtype}.txt"
    corrupted_dir = "./results/ours/imdb-corrupted=0.05.txt"
    corrupted_watermarked = []
    with open(corrupted_dir, "r") as reader:
        for line in reader:
            corrupted_watermarked.append(line)

    clean_watermarked = get_result_txt(result_dir)
    num_corrupted_sen = 0

    prev_c_idx = 0
    agg_msg = []
    bit_error_agg = {}
    infill_match_cnt = 0
    infill_match_cnt2 = 0

    num_mask_match_cnt = 0
    zero_cnt = 0
    one_cnt = 0
    kwd_match_cnt = 0
    mask_match_cnt = 0

    # agg_sentences = ""
    # agg_msg = []

    for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_text, key, msg) in enumerate(clean_watermarked):
        wm_text = corrupted_watermarked[idx].strip() if corrupted_flag else wm_text.strip()
        sen = spacy_tokenizer(wm_text.strip())

        all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
        # for sanity check, one use the original (uncorrupted) texts
        sentences_ = cover_texts[c_idx]
        sen_ = spacy_tokenizer(sentences_[sen_idx].text)
        all_keywords_, entity_keywords_ = model.keyword_module.extract_keyword([sen_])

        logger.info(f"{c_idx} {sen_idx}")

        # if attack has failed or no watermarked is embedded, skip
        if corrupted_flag and (wm_text == "skipped" or len(msg) == 0):
            continue

        num_corrupted_sen += 1
        keyword = all_keywords[0]
        ent_keyword = entity_keywords[0]
        agg_cwi, mask_idx, mask_word = model.run_iter(sen, keyword, ent_keyword, train_flag=False, embed_flag=True)
        wm_keys = model.tokenizer(" ".join([t.text for t in mask_word]), add_special_tokens=False)['input_ids']

        keyword_ = all_keywords_[0]
        ent_keyword_ = entity_keywords_[0]
        agg_cwi_, mask_idx_, mask_word_ = model.run_iter(sen_, keyword_, ent_keyword_,
                                                         train_flag=False, embed_flag=True)
        kwd_match_flag = set([x.text for x in keyword]) == set([x.text for x in keyword_])
        if kwd_match_flag:
            kwd_match_cnt += 1
        if not kwd_match_flag:
            logger.info(f"{sen}")
            logger.info(f"Extracted Kwd: {keyword}")
            logger.info(f"Original Kwd: {keyword_}")



        mask_match_flag = set(mask_idx) == set(mask_idx_)
        if mask_match_flag:
            mask_match_cnt += 1

        num_mask_match_flag = len(mask_idx) == len(mask_idx_)
        if num_mask_match_flag:
            num_mask_match_cnt += 1

        infill_match_list = []
        for a, b in zip(agg_cwi, agg_cwi_):
            infill_match_flag = (a == b).all()
            infill_match_list.append(infill_match_flag)
        if all(infill_match_list):
            infill_match_cnt += 1

        valid_watermarks = []
        valid_keys = []
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

                wm_mask = []
                wm_mask_idx = []
                for k in wm_kwd:
                    if k.head not in wm_kwd and k.head not in wm_mask and \
                            k.head not in wm_ent_kwd and not k.head.is_punct:
                        wm_mask.append(k.head)
                        wm_mask_idx.append(k.head.i)

                kwd_match_flag = set([x.text for x in wm_kwd]) == set([x.text for x in keyword])
                # checking whether the watermark can be embedded
                mask_match_flag = len(wm_mask) > 0 and set(wm_mask_idx) == set(mask_idx)
                if mask_match_flag:
                    valid_watermarks.append(wm_tokenized.text)
                    valid_keys.append(torch.stack(cwi).tolist())

        extracted_msg = []
        if len(valid_keys) > 1:
            try:
                extracted_msg_decimal = valid_keys.index(wm_keys)
                infill_match_cnt2 += 1
            except:
                similarity = [len(set(wm_keys).intersection(x)) for x in valid_keys]
                similar_key = max(zip(valid_keys, similarity), key=lambda x: x[1])[0]
                extracted_msg_decimal = valid_keys.index(similar_key)

            num_digit = math.ceil(math.log2(len(valid_keys)))
            extracted_msg = format(extracted_msg_decimal, f"0{num_digit}b")
            extracted_msg = list(map(int, extracted_msg))
        else:
            infill_match_cnt2 += 1

        one_cnt += sum(msg)
        zero_cnt += len(msg) - sum(msg)

        error_cnt, cnt = compute_ber(msg, extracted_msg)
        # if error_cnt:
        #     breakpoint()
        bit_error_agg['sentence_err_cnt'] = bit_error_agg.get('sentence_err_cnt', 0) + error_cnt
        bit_error_agg['sentence_cnt'] = bit_error_agg.get('sentence_cnt', 0) + cnt


        # logger.info(f"Corrupted sentence: {corrupted_sen}")
        # logger.info(f"original sentence: {sen}")
        # logger.info(f"Extracted msg: {' '.join(extracted_key)}")
        # logger.info(f"Gt msg: {' '.join(key)} \n")

        # agg_sentences = ""
        # agg_msg = []

    logger.info(f"Corruption Rate: {num_corrupted_sen / len(clean_watermarked):3f}")
    logger.info(f"Sentence BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']} ="
                f"{bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")
    logger.info(f"Infill match rate {infill_match_cnt/num_corrupted_sen:.3f}")
    # logger.info(f"Zero to One Ratio {zero_cnt / one_cnt:3f}")
    logger.info(f"mask match rate: {mask_match_cnt / num_corrupted_sen:.3f}")
    logger.info(f"num. mask match rate: {num_mask_match_cnt / num_corrupted_sen:.3f}")
    logger.info(f"kwd match rate: {kwd_match_cnt / num_corrupted_sen:.3f}")
    # logger.info(f"Sample BER: {bit_error_agg['sample_err_cnt'] / bit_error_agg['sample_cnt']:.3f}")

            
            
