import copy
from datetime import datetime
import logging
import os
import string
import re


from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
# from models.misc_models import tokenizer, pipe_classification, pipe_fill_mask

try:
    import yake
except:
    print("Error importing yake")

sr_threshold = 0.95
stop = set(stopwords.words('english'))
punctuation = set(string.punctuation)
riskset = stop.union(punctuation)


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
