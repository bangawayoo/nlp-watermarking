import copy
from datetime import datetime
import logging
import os
import string
import re


from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, pipeline
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

try:
    import yake
except:
    print("Error importing yake")

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


#https://www.kaggle.com/code/rowhitswami/keywords-extraction-using-tf-idf-method
def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def get_tfidf_keywords(vectorizer, feature_names, doc, topk=10):
    """Return top k keywords from a doc using TF-IDF method"""

    # generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo()) # dict of {vocab idx: score}
    result = {}
    splited_doc = doc.split(" ")
    for idx, _ in sorted_items[:topk]:
        result[splited_doc.index(feature_names[idx])] = feature_names[idx]

    return result

def get_YAKE_keywords(doc, topk=10, **kwargs):
    kw_extractor = yake.KeywordExtractor(**kwargs)
    keywords = kw_extractor.extract_keywords(doc) # dict of {phrase: score}

    result = {}
    splited_doc = doc.split(" ")

    for k in keywords[:topk]:
        word = k[0]
        if len(word.split(" ")) > 1:
            word = word.split(" ")[0]
        result[splited_doc.index(word)] = k[0]

    return result

def clean_text(text):
    PUNCTUATION = """!"#$%&'()*+,./:;<=>?@[\]^_`{|}~"""
    """Doc cleaning"""
    text = text.strip()
    # Lowering text
    text = text.lower()

    # Removing punctuation
    text = "".join([c for c in text if c not in PUNCTUATION])

    # Removing whitespace and newlines
    text = re.sub('\s+', ' ', text)

    return text


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def color_text(text, indices=[0], colorcode="OKBLUE"):
    if not isinstance(text, str):
        text = str(text)
    splited_text = text.split(' ')
    colorcode = eval(f"bcolors.{colorcode}")
    for idx in indices:
        splited_text[idx] = f"{colorcode}{splited_text[idx]}{bcolors.ENDC}"
    return " ".join(splited_text)


def find_diff_word(text, ref_text):
    """
    Currently only works when a single word is inserted
    """
    idx = 0
    for word, ref_word in zip(text.split(" "), ref_text.split(" ")):
        if word != ref_word:
            return idx, word
        idx += 1

    return None, None


def flatten_list(nested_list):
    output = []
    for l in nested_list:
        output.extend(l)


def change_str_to_int(listed_str):
    if len(listed_str) < 1:
        return
    int_str = [int(elem) for elem in listed_str if elem.isdigit()]
    return int_str



def get_result_txt(result_txt):
    results = []
    with open(f"{result_txt}", "r") as reader:
        for line in reader:
            line = line.split("\t")
            if len(line) == 7:
                # corpus idx
                line[0] = int(line[0])
                # sentence idx
                line[1] = int(line[1])
                # substituted idset
                line[2] = [change_str_to_int(listed_str.split(" ")) for
                           listed_str in line[2].split(",")[:-1]]
                # substituted index
                line[3] = change_str_to_int(line[3].split(" "))
                # watermarked text
                line[4] = line[4].strip() if len(line[4]) >0 else ""
                # substituted text
                line[5] = [x.strip() for x in line[5].split(',')]
                # embedded message
                if line[6].strip():
                    line[6] = [int(x) for x in line[6].strip().split(" ")]
                else:
                    line[6] = []
            else:
                line = ["eos"] * 7
            results.append(line)
    return results



def compute_ber(pred, gt):
    error_cnt = 0
    cnt = 0
    if len(pred) != len(gt):
        error_cnt += abs(len(gt) - len(pred))
        cnt += abs(len(gt) - len(pred))
        if len(pred) < len(gt):
            gt = gt[:len(pred)]
        else:
            pred = pred[:len(gt)]

    for p, g in zip(pred, gt):
        if p != g:
            error_cnt += 1
        cnt += 1

    return error_cnt, cnt