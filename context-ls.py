from functools import reduce
import math
import os
import string
import random

from datasets import load_dataset
from tqdm import tqdm
import torch

from config import GenericArgs
from utils.misc import compute_ber, riskset, stop
from utils.contextls_utils import synchronicity_test, substitutability_test, tokenizer, riskset, stop
from utils.logging import getLogger
from utils.dataset_utils import preprocess_txt, preprocess2sentence, get_result_txt, get_dataset
from utils.metric import Metric

random.seed(1230)

def main(cover_text, f, extracting=False):
    encoded_text = tokenizer(cover_text, add_special_tokens=False, truncation=True,
                             max_length=tokenizer.model_max_length// 2 - 2)
    word_ids = encoded_text._encodings[0].word_ids
    watermarking_wordset = [None] * len(encoded_text['input_ids'])
    substituted_idset = []
    substituted_indices = []
    watermarked_text = []
    message = []

    latest_embed_index = -1
    index = 1
    while index < len(encoded_text['input_ids']) - f:
        text = tokenizer.decode(encoded_text['input_ids'][index])
        watermarked_text.append(text)
        if text in riskset:
            watermarking_wordset[index] = 'riskset'
            index = index + 1
            continue

        valid_indx = [t == index for t in word_ids]
        if sum(valid_indx) >= 2:  # skip this word; subword
            watermarking_wordset[index] = 'subword'
            index = index + 1
            continue

        local_context = encoded_text['input_ids'][:index + 2]
        is_synch, words = synchronicity_test(index, local_context)

        if not is_synch:  # skip this word if synchronicity test fails
            watermarking_wordset[index] = 'syn'
            index = index + 1
            continue

        if index - latest_embed_index != f + 1:
            if not substitutability_test(local_context[:-1], index,
                                         words):  # skip this word if substitutability test fails
                watermarking_wordset[index] = 'sub'
                index = index + 1
                continue

        watermarking_wordset[index] = tokenizer.decode(words)
        words_decoded = tokenizer.decode(words).split(" ")
        # sort token_id alphabetically
        words = [w for w, wd in sorted(zip(words, words_decoded), key=lambda pair: pair[1])]
        words_decoded.sort()

        # extraction
        if extracting:
            extracted_msg = words_decoded.index(text)
            message.append(extracted_msg)

        # embedding
        else:
            random_msg = random.choice([0,1])
            word_chosen_by_msg = words[random_msg]
            encoded_text['input_ids'][index] = word_chosen_by_msg
            message.append(random_msg)


        substituted_idset.append(words)
        substituted_indices.append(index)
        latest_embed_index = index
        index = index + f + 1

    return substituted_idset, substituted_indices, watermarking_wordset, encoded_text['input_ids'], message



if __name__ == "__main__":
    parser = GenericArgs()
    args = parser.parse_args()
    DEBUG_MODE = args.debug_mode
    if args.embed and args.extract:
        mode = "BOTH"
    elif args.embed:
        mode = "EMBED"
    elif args.extract:
        mode = "EXTRACT"
        if args.extract_corrupted:
            corrupted_dir = args.corrupted_file_dir
            corruption_type = os.path.basename(corrupted_dir).split("-")[1]
            mode = f"EXTRACT-{corruption_type}"
    else:
        raise AssertionError("Mode should be specified")

    dirname = f"./results/context-ls/{args.dtype}/{args.exp_name}"
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    logger = getLogger(f"{mode}",
                       dir_=dirname,
                       debug_mode=DEBUG_MODE)
    sr_score = []
    device = torch.device("cuda")
    metric = Metric(device, **vars(args))

    f = 1
    dtype = args.dtype
    start_sample_idx = 0

    _, corpus, num_sample2load = get_dataset(dtype)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    result_dir = os.path.join(dirname, "watermarked.txt")

    cover_texts = preprocess_txt(corpus)
    cover_texts = preprocess2sentence(cover_texts, dtype+"-test", start_sample_idx, num_sample2load['test'],
                                      spacy_model=args.spacy_model)

    num_sentence = 0
    for c_idx, sentences in enumerate(cover_texts):
        num_sentence += len(sentences)
        if num_sentence >= args.num_sample:
            break
    cover_texts = cover_texts[:c_idx + 1]

    bit_count = 0
    word_count = 0

    if args.metric_only:
        for model_name in ["roberta", "all-MiniLM-L6-v2"]:
            ss_score, ss_dist = metric.compute_ss(result_dir, model_name, cover_texts)
            logger.info(f"ss score: {sum(ss_score) / len(ss_score):.3f}")
        nli_score = metric.compute_nli(result_dir, cover_texts)
        logger.info(f"nli score: {sum(nli_score) / len(nli_score):.3f}")
        exit()

    if args.embed:
        # assert not os.path.isfile(result_dir), f"{result_dir} already exists!"
        wr = open(result_dir, "w")
        for c_idx, sentences in enumerate(tqdm(cover_texts)):
            for sen_idx, sen in enumerate(sentences):
                sen_text = sen.text
                logger.info(sen)
                substituted_idset, substituted_indices, watermarking_wordset, encoded_text, message = main(sen_text, f)
                punct_removed = sen_text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
                word_count += len([i for i in punct_removed.split(" ") if i not in stop])
                num_cases = 1
                for sid in substituted_idset:
                    num_cases *= len(sid)
                bit_count += math.log2(num_cases)
                s_idset_str = ""
                for s_id in substituted_idset:
                    s_idset_str += " ".join(str(x) for x in s_id) + ","
                s_indices_str = " ".join(str(x) for x in substituted_indices)
                message_str = [str(m) for m in message]
                message_str = " ".join(message_str) if len(message_str) else ""
                watermarked_text = tokenizer.decode(encoded_text)
                keys = [tokenizer.decode(s_id) for s_id in substituted_idset]
                keys_str = ", ".join(keys)
                wr.write(f"{c_idx}\t{sen_idx}\t{s_idset_str}\t{s_indices_str}\t"
                         f"{watermarked_text}\t{keys_str}\t{message_str}\n")
            if word_count > 0:
                logger.info(f"Sample {c_idx} bpw={bit_count / word_count:.3f}")

        wr.close()


        logger.info(f"Bpw: {bit_count / word_count:.3f}")
        ss_scores = []
        for model_name in ["roberta", "all-MiniLM-L6-v2"]:
            ss_score, ss_dist = metric.compute_ss(result_dir, model_name, cover_texts)
            ss_scores.append(sum(ss_score) / len(ss_score))
            logger.info(f"ss score: {sum(ss_score) / len(ss_score):.3f}")
        nli_score = metric.compute_nli(result_dir, cover_texts)
        logger.info(f"nli score: {sum(nli_score) / len(nli_score):.3f}")

        with open(os.path.join(dirname, "embed-metrics.txt"), "a") as wr:
            wr.write(f"num.sample={args.num_sample}\t bpw={bit_count / word_count}\t "
                     f"ss={ss_scores[0]}\t ss={ss_scores[1]}\t"
                     f"nli={sum(nli_score) / len(nli_score)}\n")


    ##
    if args.extract:
        corrupted_flag = args.extract_corrupted
        corrupted_dir = args.corrupted_file_dir

        corrupted_watermarked = []
        if corrupted_flag:
            with open(corrupted_dir, "r") as reader:
                for line in reader:
                    line = line.split("[sep] ")
                    corrupted_watermarked.append(line)

        clean_watermarked = get_result_txt(result_dir)

        num_corrupted_sen = 0
        sample_level_bit = {'gt':[], 'extracted':[]}
        bit_error_agg = {}
        midx_match_cnt = 0
        mword_match_cnt = 0
        infill_match_cnt = 0
        prev_c_idx = 0

        for idx, (c_idx, sen_idx, sub_idset, sub_idx, clean_wm_sen, key, msg) in enumerate(clean_watermarked):
            if prev_c_idx != c_idx:
                error_cnt, cnt = compute_ber(sample_level_bit['extracted'], sample_level_bit['gt'])
                bit_error_agg['sample_err_cnt'] = bit_error_agg.get('sample_err_cnt', 0) + error_cnt
                bit_error_agg['sample_cnt'] = bit_error_agg.get('sample_cnt', 0) + cnt
                sample_level_bit = {'gt': [], 'extracted': []}
                prev_c_idx = c_idx

            if len(corrupted_watermarked) > 0 and len(corrupted_watermarked) <= idx:
                logger.info(f"Extracted all corrupted watermarks. Ending extraction... ")
                break

            original_sentences = cover_texts[c_idx]
            sen = original_sentences[sen_idx]

            clean_encoded = tokenizer(clean_wm_sen.strip(), add_special_tokens=False, truncation=True,
                                      max_length=tokenizer.model_max_length// 2 - 2)['input_ids']
            wm_texts = corrupted_watermarked[idx] if corrupted_flag else [clean_wm_sen.strip()]
            for wm_text in wm_texts:
                wm_text = wm_text.strip()
                if corrupted_flag and wm_text == "skipped":
                    continue

                num_corrupted_sen += 1
                extracted_idset, extracted_indices, watermarking_wordset, encoded_text, extracted_msg = \
                    main(wm_text, f, extracting=True)
                extracted_key = [tokenizer.decode(s_id) for s_id in extracted_idset]

                midx_match_flag = set(extracted_indices) == set(sub_idx)
                if midx_match_flag:
                    midx_match_cnt += 1

                mword_match_flag = set([encoded_text[idx] for idx in extracted_indices]) == \
                                   set([clean_encoded[idx] for idx in sub_idx])
                if mword_match_flag:
                    mword_match_cnt += 1

                infill_match_list = []
                if len(sub_idset) == len(extracted_idset):
                    for a, b in zip(sub_idset, extracted_idset):
                        infill_match_flag = a==b
                        infill_match_list.append(infill_match_flag)
                else:
                    infill_match_list.append(False)
                if all(infill_match_list):
                    infill_match_cnt += 1

                sample_level_bit['extracted'].extend(extracted_msg)
                sample_level_bit['gt'].extend(msg)
                error_cnt, cnt = compute_ber(msg, extracted_msg)
                bit_error_agg['sentence_err_cnt'] = bit_error_agg.get('sentence_err_cnt', 0) + error_cnt
                bit_error_agg['sentence_cnt'] = bit_error_agg.get('sentence_cnt', 0) + cnt

                match_flag = error_cnt == 0
                logger.debug(f"{c_idx} {sen_idx} {match_flag}")
                logger.info(f"Corrupted sentence: {wm_text}")
                logger.info(f"original sentence: {sen}")
                logger.info(f"Extracted msg: {' '.join(extracted_key)}")
                logger.info(f"Gt msg: {' '.join(key)} \n")


        # logger.info(f"Corruption Rate: {num_corrupted_sen / len(clean_watermarked):3f}")
        logger.info(f"Sentence BER: {bit_error_agg['sentence_err_cnt']}/{bit_error_agg['sentence_cnt']}="
                    f"{bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")

        logger.info(f"mask infill match rate: {infill_match_cnt / num_corrupted_sen:.3f}")
        logger.info(f"mask index match rate: {midx_match_cnt / num_corrupted_sen:.3f}")
        logger.info(f"mask word match rate: {mword_match_cnt / num_corrupted_sen:.3f}")

        if corrupted_flag:
            with open(os.path.join(dirname, "ber.txt"), "a") as wr:
                wr.write(f"{corrupted_dir}\t {bit_error_agg['sentence_err_cnt']/bit_error_agg['sentence_cnt']}\n")