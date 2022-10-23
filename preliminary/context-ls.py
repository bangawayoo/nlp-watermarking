#TODO:
# 1. check bpw on other datasets


from transformers import pipeline
from transformers import AutoTokenizer
import string

from textattack import attack_recipes

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from utils import *

lemmatizer = WordNetLemmatizer()

from IPython.display import display

stop = set(stopwords.words('english'))
puncuation = set(string.punctuation)
riskset = stop.union(puncuation)


def synchronicity_test(index, local_context):
    mask_candidates = generate_substitute_candidates(local_context)

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


def generate_substitute_candidates(text_processed):
    text_for_ls = concatenate_for_ls(text_processed)
    mask_candidates = pipe_fill_mask(text_for_ls)

    # filter out words with only difference in cases (lowercase, uppercase)
    text = tokenizer.decode(text_processed[-2])
    mask_candidates = list(filter(lambda x: not (x['token_str'].lower() == text.lower() and
                                            x['token_str'] != text) , mask_candidates))
    # mask_candidates = list(filter(lambda x: x['token_str'] != text, mask_candidates))

    # filter out morphorlogical derivations
    text_lm = lemmatizer.lemmatize(text, 'v')
    mask_candidates = \
        list(filter(lambda x: lemmatizer.lemmatize(x['token_str'], 'v') != text_lm
                              or x['token_str']==text, mask_candidates))

    # filter out riskset
    mask_candidates = list(filter(lambda x: x['token_str'] not in riskset, mask_candidates))

    # filter out subword tokens
    mask_candidates = list(filter(lambda x: not x['token_str'].startswith("##"), mask_candidates))

    for item in mask_candidates:
        replaced = text_for_ls.replace('[MASK]', item['token_str'])
        entail_result = pipe_classification(replaced)
        item['entail_score'] = [i for i in entail_result[0] if i['label']=='ENTAILMENT'][0]['score']

    # sort in descending order
    mask_candidates = sorted(mask_candidates, key=lambda x: x['entail_score'], reverse=True)

    # filter out with sr_threshold
    mask_candidates = list(filter(lambda x: x['entail_score'] > sr_threshold, mask_candidates))

    return mask_candidates[:2]  # top-2 words


def main(cover_text, f):
    encoded_text = tokenizer(cover_text, add_special_tokens=False, truncation=True)
    word_ids = encoded_text._encodings[0].word_ids
    watermarking_wordset = [None] * len(encoded_text['input_ids'])
    substituted_idset = []
    substituted_indices = []
    watermarked_text = []

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
        substituted_idset.append(words)
        substituted_indices.append(index)
        latest_embed_index = index
        index = index + f + 1

    return substituted_idset, substituted_indices, watermarking_wordset, encoded_text


##

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", is_split_into_words=True)
pipe_fill_mask = pipeline('fill-mask', model='bert-base-cased', device=0, top_k=32)
pipe_classification = pipeline(model="roberta-large-mnli", device=0, top_k=None)


##
import itertools
from datasets import arxiv_cs_abstracts, roc_stories
import sys

f = 1
sr_threshold = 0.95
corpus = arxiv_cs_abstracts("train",  attrs=['abstract'])
result_dir = "results/context-ls-result.txt"

cover_texts = corpus[:10]
cover_texts = [t.replace("\n", " ") for t in cover_texts]

if sys.argv[1] == "embed":
    # assert not os.path.isfile(result_dir), f"{result_dir} already exists!"
    wr = open("results/context-ls-result.txt", "a")
    for c_idx, cover_text in enumerate(cover_texts):
        sentences = nltk.sent_tokenize(cover_text)
        for sen_idx, sen in enumerate(sentences):
            print(sen)
            substituted_idset, substituted_indices, watermarking_wordset, encoded_text = main(sen, f)
            s_idset_str = ""
            for s_id in substituted_idset:
                s_idset_str += " ".join(str(x) for x in s_id) + ","
            s_indices_str = " ".join(str(x) for x in substituted_indices)
            keys = [tokenizer.decode(s_id) for s_id in substituted_idset]
            keys_str = ", ".join(keys)
            print(f"{c_idx}\t{sen_idx}\t{s_idset_str}\t{s_indices_str}\t{keys_str}\n")
            wr.write(f"{c_idx}\t{sen_idx}\t{s_idset_str}\t{s_indices_str}\t{keys_str}\n")

    wr.close()


##

if sys.argv[1] == "extract":
    pct = 0.001
    watermarked = []
    # with open(f"./results/corrupted={pct}-watermarked.txt", "r") as reader:
    with open(f"./results/context-ls-result.txt", "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                watermarked.append(line)

    with open("results/context-ls-keys.txt", "r") as reader:
        keys = []
        for line in reader:
            line = line.strip()
            if line:
                keys.append(line)

    match = 0

    for c_idx, w_text in enumerate(watermarked):
        substituted_idset, substituted_indices, watermarking_wordset, encoded_text = main(w_text, f)
        wm_key = [tokenizer.decode(s_id) for s_id in substituted_idset]
        idx, inserted_word = find_diff_word(w_text, cover_texts[c_idx])
        print(color_text(w_text, [idx]))
        print("Extracted msg:", end="  ")
        print(wm_key)
        print("Gt msg", end=" ")
        print(keys[c_idx])
        print("\n")


