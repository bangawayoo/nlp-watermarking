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
from tqdm.auto import tqdm

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
from utils.dataset_utils import get_result_txt
from utils.logging import getLogger
from utils.metric import Metric
from utils.misc import compute_ber

random.seed(1230)

infill_parser = WatermarkArgs()
generic_parser = GenericArgs()
infill_args, _ = infill_parser.parse_known_args()
generic_args, _ = generic_parser.parse_known_args()
DEBUG_MODE = generic_args.debug_mode
dtype = generic_args.dtype

dirname = f"./results/ours/{dtype}/{generic_args.exp_name}"
start_sample_idx = 0
num_sample = generic_args.num_sample

spacy_tokenizer = spacy.load(generic_args.spacy_model)
if "trf" in generic_args.spacy_model:
    spacy.require_gpu()
model = InfillModel(infill_args, dirname=dirname)

_, cover_texts = model.return_dataset()

num_sentence = 0
for c_idx, sentences in enumerate(cover_texts):
    num_sentence += len(sentences)
    if num_sentence >= num_sample:
        break
cover_texts = cover_texts[:c_idx+1]

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

if generic_args.embed:
    logger = getLogger("EMBED",
                       dir_=dirname,
                       debug_mode=DEBUG_MODE)



    result_dir = os.path.join(dirname, "watermarked.txt")
    if not os.path.exists(result_dir):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    if generic_args.metric_only:
        for model_name in ["roberta", "all-MiniLM-L6-v2"]:
            ss_score, ss_dist = metric.compute_ss(result_dir, model_name, cover_texts)
            logger.info(f"ss score: {sum(ss_score) / len(ss_score):.3f}")
        nli_score = metric.compute_nli(result_dir, cover_texts)
        logger.info(f"nli score: {sum(nli_score) / len(nli_score):.3f}")
        exit()

    progress_bar = tqdm(range(len(cover_texts)))
    wr = open(result_dir, "w")
    for c_idx, sentences in enumerate(cover_texts):
        for s_idx, sen in enumerate(sentences):
            sen = spacy_tokenizer(sen.text.strip())
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
            logger.info(f"{c_idx} {s_idx}")
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
                        valid_watermarks.append(wm_tokenized.text)
                        mask_match_cnt += 1

                    sample_cnt += 1
                    candidate_kwd_cnt += 1
    
            punct_removed = sen.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])
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

                keys = []
                wm_tokenized = spacy_tokenizer(wm_text)
                for m_idx in mask_idx:
                    keys.append(wm_tokenized[m_idx].text)
                keys_str = ", ".join(keys)
                message_str = ' '.join(message_str) if len(message_str) else ""
                wr.write(f"{c_idx}\t{s_idx}\t \t \t"
                         f"{''.join(wm_text)}\t{keys_str}\t{message_str}\n")
            else:
                original_text = ''.join(tokenized_text)
                wr.write(f"{c_idx}\t{s_idx}\t \t \t"
                         f"{original_text}\t \t \n")

            if candidate_kwd_cnt > 0:
                upper_bound += math.log2(candidate_kwd_cnt)
        progress_bar.update(1)
        if word_count:
            logger.info(f"Bpw : {bit_count / word_count:.3f}")

    wr.close()
    logger.info(infill_args)
    logger.info(f"UB Bpw : {upper_bound / word_count:.3f}")
    logger.info(f"Bpw : {bit_count / word_count:.3f}")
    assert sample_cnt > 0, f"No candidate watermarked sets were created"
    logger.info(f"mask match rate: {mask_match_cnt / sample_cnt:.3f}")
    logger.info(f"kwd match rate: {kwd_match_cnt / sample_cnt :.3f}")
    logger.info(f"zero/one ratio: {zero_cnt / one_cnt :.3f}")

    ss_scores = []
    for model_name in ["roberta", "all-MiniLM-L6-v2"]:
        ss_score, ss_dist = metric.compute_ss(result_dir, model_name, cover_texts)
        ss_scores.append(sum(ss_score) / len(ss_score))
        logger.info(f"ss score: {sum(ss_score) / len(ss_score):.3f}")
    nli_score = metric.compute_nli(result_dir, cover_texts)
    logger.info(f"nli score: {sum(nli_score) / len(nli_score):.3f}")

    result_dir = os.path.join(dirname, "embed-metrics.txt")
    with open(result_dir, "a") as wr:
        wr.write(str(vars(infill_args))+"\n")
        wr.write(f"num.sample={num_sample}\t bpw={bit_count / word_count}\t "
                 f"ss={ss_scores[0]}\t ss={ss_scores[1]}\t"
                 f"nli={sum(nli_score) / len(nli_score)}\n")

      
      
##

if generic_args.extract:
    corrupted_flag = generic_args.extract_corrupted
    logger_name = "EXTRACT"
    if corrupted_flag:
        corrupted_dir = generic_args.corrupted_file_dir
        corruption_type = os.path.basename(corrupted_dir).split("-")[1]
        logger_name = f"EXTRACT-{corruption_type}"
    logger = getLogger(logger_name,
                       dir_=dirname, debug_mode=DEBUG_MODE)
    start_sample_idx = 0

    result_dir = os.path.join(dirname, "watermarked.txt")
    if corrupted_flag:
        logger.info(f"Extracting corrupted watermarks on {corrupted_dir}...")
        corrupted_watermarked = []
        with open(corrupted_dir, "r") as reader:
            for line in reader:
                line = line.split("[sep] ")
                corrupted_watermarked.append(line)

    clean_watermarked = get_result_txt(result_dir)
    num_corrupted_sen = 0

    prev_c_idx = 0
    agg_msg = []
    bit_error_agg = {"sentence_err_cnt": 0, "sentence_cnt": 0}
    infill_match_cnt = 0
    infill_match_cnt2 = 0

    num_mask_match_cnt = 0
    zero_cnt = 0
    one_cnt = 0
    kwd_match_cnt = 0
    midx_match_cnt = 0
    mword_match_cnt = 0

    # agg_sentences = ""
    # agg_msg = []

    for idx, (c_idx, sen_idx, sub_idset, sub_idx, clean_wm_text, key, msg) in enumerate(clean_watermarked):
        try:
            wm_texts = corrupted_watermarked[idx] if corrupted_flag else [clean_wm_text.strip()]
        except IndexError:
            logger.debug("Create corrupted samples is less than watermarked. Ending extraction...")
            break
        for wm_text in wm_texts:
            num_corrupted_sen += 1
            sen = spacy_tokenizer(wm_text.strip())
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
            # for sanity check, one use the original (uncorrupted) watermarked texts
            sen_ = spacy_tokenizer(clean_wm_text.strip())
            all_keywords_, entity_keywords_ = model.keyword_module.extract_keyword([sen_])

            logger.info(f"{c_idx} {sen_idx}")
            # extracting states for corrupted
            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]
            agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword, ent_keyword,
                                                                                                  train_flag=False, embed_flag=True)
            wm_keys = model.tokenizer(" ".join([t.text for t in mask_word]), add_special_tokens=False)['input_ids']

            # extracting states for uncorrupted
            keyword_ = all_keywords_[0]
            ent_keyword_ = entity_keywords_[0]
            agg_cwi_, agg_probs_, tokenized_pt_, (mask_idx_pt_, mask_idx_, mask_word_) = model.run_iter(sen_, keyword_, ent_keyword_,
                                                                                                        train_flag=False, embed_flag=True)
            kwd_match_flag = set([x.text for x in keyword]) == set([x.text for x in keyword_])
            if kwd_match_flag:
                kwd_match_cnt += 1

            midx_match_flag = set(mask_idx) == set(mask_idx_)
            if midx_match_flag:
                midx_match_cnt += 1

            mword_match_flag = set([m.text for m in mask_word]) == set([m.text for m in mask_word_])
            if mword_match_flag:
                mword_match_cnt += 1

            num_mask_match_flag = len(mask_idx) == len(mask_idx_)
            if num_mask_match_flag:
                num_mask_match_cnt += 1

            infill_match_list = []
            if len(agg_cwi) == len(agg_cwi_):
                for a, b in zip(agg_cwi, agg_cwi_):
                    if len(a) == len(b):
                        infill_match_flag = (a == b).all()
                    else:
                        infill_match_flag = False
                    infill_match_list.append(infill_match_flag)
            else:
                infill_match_list.append(False)
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
                    wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)

                    # checking whether the watermark can be embedded
                    mask_match_flag = len(wm_mask) > 0 and set(wm_mask_idx) == set(mask_idx)
                    if mask_match_flag:
                        valid_watermarks.append(wm_tokenized.text)
                        valid_keys.append(torch.stack(cwi).tolist())

            extracted_msg = []
            if len(valid_keys) > 1:
                try:
                    extracted_msg_decimal = valid_keys.index(wm_keys)
                except:
                    similarity = [len(set(wm_keys).intersection(x)) for x in valid_keys]
                    similar_key = max(zip(valid_keys, similarity), key=lambda x: x[1])[0]
                    extracted_msg_decimal = valid_keys.index(similar_key)

                num_digit = math.ceil(math.log2(len(valid_keys)))
                extracted_msg = format(extracted_msg_decimal, f"0{num_digit}b")
                extracted_msg = list(map(int, extracted_msg))

            one_cnt += sum(msg)
            zero_cnt += len(msg) - sum(msg)

            error_cnt, cnt = compute_ber(msg, extracted_msg)
            bit_error_agg['sentence_err_cnt'] = bit_error_agg.get('sentence_err_cnt', 0) + error_cnt
            bit_error_agg['sentence_cnt'] = bit_error_agg.get('sentence_cnt', 0) + cnt

            # logger.info(f"Corrupted sentence: {corrupted_sen}")
            # logger.info(f"original sentence: {sen}")
            # logger.info(f"Extracted msg: {' '.join(extracted_key)}")
            # logger.info(f"Gt msg: {' '.join(key)} \n")
            # agg_sentences = ""
            # agg_msg = []
        if bit_error_agg['sentence_cnt']:
            logger.info(f"BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']}="
                        f"{bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")

    if corrupted_flag:
        # logger.info(f"Corruption Rate: {num_corrupted_sen / len(corrupted_watermarked):3f}")
        with open(os.path.join(dirname, "ber.txt"), "a") as wr:
            wr.write(f"{corrupted_dir}\t {bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']}\n")

    logger.info(f"Sentence BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']}="
                f"{bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")
    logger.info(f"Infill match rate {infill_match_cnt/num_corrupted_sen:.3f}")
    # logger.info(f"Zero to One Ratio {zero_cnt / one_cnt:3f}")
    logger.info(f"mask index match rate: {midx_match_cnt / num_corrupted_sen:.3f}")
    logger.info(f"mask word match rate: {mword_match_cnt / num_corrupted_sen:.3f}")
    logger.info(f"kwd match rate: {kwd_match_cnt / num_corrupted_sen:.3f}")
    logger.info(f"num. mask match rate: {num_mask_match_cnt / num_corrupted_sen:.3f}")

    # logger.info(f"Sample BER: {bit_error_agg['sample_err_cnt'] / bit_error_agg['sample_cnt']:.3f}")

            
            
