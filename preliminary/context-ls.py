#TODO:
import logging
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from collections import defaultdict
import random
import sys
import string

from transformers import pipeline
from transformers import AutoTokenizer

from textattack import attack_recipes
from tqdm import tqdm

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import nltk

from utils import *


lemmatizer = WordNetLemmatizer()

from IPython.display import display

stop = set(stopwords.words('english'))
punctuation = set(string.punctuation)
riskset = stop.union(punctuation)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(f"./logs/{sys.argv[-1]}.log")
file_handler.setFormatter(logFormatter)
logger.addHandler(file_handler)


def synchronicity_test(index, local_context):
    mask_candidates = generate_substitute_candidates(local_context)

    if local_context[-2] not in [word['token'] for word in mask_candidates]:
        return False, None

    if len(mask_candidates) < 2:  # skip this word; do not exist appropriate candidates
        return False, None

    top1_processed = tokenizer(mask_candidates[0]['sequence'], add_special_tokens=False)['input_ids'][index + 2:]
    top2_processed = tokenizer(mask_candidates[1]['sequence'], add_special_tokens=False)['input_ids'][index + 2:]

    mask_candidates1 = generate_substitute_candidates(top1_processed)
    mask_candidates2 = generate_substitute_candidates(top2_processed)

    FC = [word['token_str'] for word in mask_candidates]
    FC1 = [word['token_str'] for word in mask_candidates1]
    FC2 = [word['token_str'] for word in mask_candidates2]
    return set(FC) == set(FC1) == set(FC2), [word['token'] for word in mask_candidates]


def substitutability_test(new_context, index, words):
    for w in words:
        new_context[-1] = w
        is_synch, _ = synchronicity_test(index - 1, new_context,)
        if is_synch:
            return False
    return True


def concatenate_for_ls(text):
    masked_text = text.copy()
    masked_text[-2] = tokenizer.mask_token_id
    tokens_for_ls = text + [tokenizer.sep_token_id] + masked_text
    return tokenizer.decode(tokens_for_ls)


def generate_substitute_candidates(text_processed, topk=2):
    text_for_ls = concatenate_for_ls(text_processed)
    mask_candidates = pipe_fill_mask(text_for_ls)

    # filter out words with only difference in cases (lowercase, uppercase)
    text = tokenizer.decode(text_processed[-2])
    mask_candidates = list(filter(lambda x: not (x['token_str'].lower() == text.lower() and
                                            x['token_str'] != text) , mask_candidates))

    # filter out subword tokens
    mask_candidates = list(filter(lambda x: not x['token_str'].startswith("##"), mask_candidates))

    # filter out morphological derivations
    porter_stemmer = PorterStemmer()
    text_lm = porter_stemmer.stem(text)
    mask_candidates = \
        list(filter(lambda x: porter_stemmer.stem(x['token_str']) != text_lm
                              or x['token_str']==text, mask_candidates))

    lemmatizer = WordNetLemmatizer()
    for pos in ["v", "n", "a", "r", "s"]:
        text_lm = lemmatizer.lemmatize(text, pos)
        mask_candidates = \
            list(filter(lambda x: lemmatizer.lemmatize(x['token_str'], pos) != text_lm
                                  or x['token_str']==text, mask_candidates))
    # filter out riskset
    mask_candidates = list(filter(lambda x: x['token_str'] not in riskset, mask_candidates))
    # filter out words with any punctuations
    mask_candidates = list(filter(lambda x: not any(s for s in x['token_str'] if s in string.punctuation),
                                  mask_candidates))


    # get entailment scores
    for item in mask_candidates:
        replaced = text_for_ls.replace('[MASK]', item['token_str'])
        entail_result = pipe_classification(replaced)
        item['entail_score'] = [i for i in entail_result[0] if i['label']=='ENTAILMENT'][0]['score']

    # sort in descending order
    mask_candidates = sorted(mask_candidates, key=lambda x: x['entail_score'], reverse=True)

    # filter out with sr_threshold
    mask_candidates = list(filter(lambda x: x['entail_score'] > sr_threshold, mask_candidates))

    return mask_candidates[:topk]


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

    return substituted_idset, substituted_indices, watermarking_wordset, encoded_text, message


##

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
pipe_fill_mask = pipeline('fill-mask', model='bert-base-cased', device=0, top_k=32)
pipe_classification = pipeline(model="roberta-large-mnli", device=0, top_k=None)


##
from datasets import load_dataset
from dataset_utils import arxiv_cs_abstracts, roc_stories, preprocess_imdb, preprocess2sentence
import sys
import spacy


f = 1
sr_threshold = 0.95
dtype = "imdb"
start_sample_idx = 0
num_sample = 100
# corpus = arxiv_cs_abstracts("train",  attrs=['abstract'])
corpus = load_dataset("imdb")['test']['text']
result_dir = f"results/context-ls-{dtype}-{start_sample_idx}-{start_sample_idx+num_sample}.txt"

cover_texts = [t.replace("\n", " ") for t in corpus]
cover_texts = preprocess_imdb(cover_texts)
cover_texts = preprocess2sentence(cover_texts, start_sample_idx, num_sample)

bit_count = 0
word_count = 0


if sys.argv[1] == "embed":
    # assert not os.path.isfile(result_dir), f"{result_dir} already exists!"
    wr = open(result_dir, "w")
    for c_idx, sentences in enumerate(tqdm(cover_texts)):
        for sen_idx, sen in enumerate(sentences):
            logger.info(sen)
            substituted_idset, substituted_indices, watermarking_wordset, encoded_text, message = main(sen, f)
            punct_removed = sen.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])
            bit_count += len(substituted_idset)

            s_idset_str = ""
            for s_id in substituted_idset:
                s_idset_str += " ".join(str(x) for x in s_id) + ","
            s_indices_str = " ".join(str(x) for x in substituted_indices)
            message_str = [str(m) for m in message]
            message_str = " ".join(message_str) if len(message_str) else ""
            watermarked_text = tokenizer.decode(encoded_text['input_ids'])
            keys = [tokenizer.decode(s_id) for s_id in substituted_idset]
            keys_str = ", ".join(keys)
            wr.write(f"{c_idx}\t{sen_idx}\t{s_idset_str}\t{s_indices_str}\t"
                     f"{watermarked_text}\t{keys_str}\t{message_str}\n")
        logger.info(f"Sample{c_idx} bpw={bit_count / word_count:.3f}")

    wr.close()

    with open("results/bpw.txt", "a") as wr:
        wr.write(f"{dtype}-{num_sample}\t {bit_count/word_count}\n")

##
import random

if sys.argv[1] == "extract":
    dtype = "imdb"
    start_sample_idx = 0
    num_sample = 100
    result_dir = f"results/context-ls-{dtype}-{start_sample_idx}-{start_sample_idx+num_sample}.txt"
    corrupted_flag = True
    corrupted_dir = "./results/context-ls-imdb-corrupted=0.1.txt"
    corrupted_watermarked = []
    with open(corrupted_dir, "r") as reader:
        for line in reader:
            corrupted_watermarked.append(line)

    clean_watermarked = get_result_txt(result_dir)

    num_corrupted_sen = 0
    sample_level_bit = {'gt':[], 'extracted':[]}
    bit_error_agg = {}

    prev_c_idx = 0
    for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_sen, key, msg) in enumerate(clean_watermarked):
        if prev_c_idx != c_idx:
            error_cnt, cnt = compute_ber(sample_level_bit['extracted'], sample_level_bit['gt'])
            bit_error_agg['sample_err_cnt'] = bit_error_agg.get('sample_err_cnt', 0) + error_cnt
            bit_error_agg['sample_cnt'] = bit_error_agg.get('sample_cnt', 0) + cnt
            sample_level_bit = {'gt':[], 'extracted':[]}
            prev_c_idx = c_idx

        original_sentences = cover_texts[c_idx]
        sen = original_sentences[sen_idx]
        corrupted_sen = corrupted_watermarked[idx].strip() if corrupted_flag else wm_sen

        if corrupted_flag and (corrupted_sen == "skipped" or len(msg) == 0):
            continue

        num_corrupted_sen += 1
        extracted_idset, extracted_indices, watermarking_wordset, encoded_text, message = \
            main(corrupted_sen, f, extracting=True)
        extracted_key = [tokenizer.decode(s_id) for s_id in extracted_idset]

        sample_level_bit['extracted'].extend(message)
        sample_level_bit['gt'].extend(msg)
        error_cnt, cnt = compute_ber(msg, message)
        bit_error_agg['sentence_err_cnt'] = bit_error_agg.get('sentence_err_cnt', 0) + error_cnt
        bit_error_agg['sentence_cnt'] = bit_error_agg.get('sentence_cnt', 0) + cnt

        match_flag = error_cnt == 0
        logger.info(f"{c_idx} {sen_idx} {match_flag}")
        logger.info(f"Corrupted sentence: {corrupted_sen}")
        logger.info(f"original sentence: {sen}")
        logger.info(f"Extracted msg: {' '.join(extracted_key)}")
        logger.info(f"Gt msg: {' '.join(key)} \n")


    logger.info(f"Corruption Rate: {num_corrupted_sen / len(clean_watermarked):3f}")
    logger.info(f"Sample BER: {bit_error_agg['sample_err_cnt'] / bit_error_agg['sample_cnt']:.3f}")
    logger.info(f"Sentence BER: {bit_error_agg['sentence_err_cnt'] / bit_error_agg['sentence_cnt']:.3f}")

    with open("./results/ber.txt", "a") as wr:
        wr.write(f"{dtype}-{num_sample}\t {bit_error_agg['sentence_err_cnt']/bit_error_agg['sentence_cnt']}\t"
                 f"{bit_error_agg['sample_err_cnt']/bit_error_agg['sample_cnt']}"
                 f"\n")