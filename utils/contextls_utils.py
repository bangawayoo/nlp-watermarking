import copy
from datetime import datetime
import logging
import os
import string
import re


from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from transformers import AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
pipe_fill_mask = pipeline('fill-mask', model='bert-base-cased', device=0, top_k=32)
pipe_classification = pipeline(model="roberta-large-mnli", device=0, top_k=None)
sr_threshold = 0.95
stop = set(stopwords.words('english'))
punctuation = set(string.punctuation)
riskset = stop.union(punctuation)

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