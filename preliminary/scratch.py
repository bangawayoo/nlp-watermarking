##
dtype = "dracula"
with open(f"./data/{dtype}.txt", "r") as f:
    text = f.readlines()

text_string = " ".join(text).replace("\n", "")
text_list = text_string.split("  ")
split_point = int(0.4 * len(text_list))
corpus = text_list[:split_point]
test_corpus = text_list[split_point:]

with open(f"./data/train-{dtype}.txt", "w") as f:
    for line in corpus:
        f.write(f"{line}\n")

##
yake_kwargs = {'lan': "en",
               "n": 1,
               "dedupLim": 0.9,
               "dedupFunc": 'seqm',
               "windowsSize": 1,
               }

import yake
from datasets import load_dataset
corpus = load_dataset('imdb')['train']['text']
sample = corpus[0]
kw_extractor = yake.KeywordExtractor(**yake_kwargs, top=20)
keywords = kw_extractor.extract_keywords(sample)  # dict of {phrase: score}
keywords

for pn in PN:
    print(pn)
    print(list(pn.children), end=" ")
    print(pn.head, end="\n\n")

##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.chdir(os.path.join(os.getcwd(), "./preliminary"))
from preliminary.dataset_utils import arxiv_cs_abstracts, roc_stories

corpus = arxiv_cs_abstracts("train", attrs=['title', 'abstract'])
corpus = [i.replace("\n", " ") for i in corpus]

# corpus = roc_stories("test")
# corpus = [i.replace("\n", " ") for i in corpus]

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from preliminary.utils import *

max_corpus_sample = len(corpus)

stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(lowercase=True, stop_words=stop_words)

processed_corpus = [clean_text(a) for a in corpus[:max_corpus_sample]]
vectorizer.fit_transform(processed_corpus)
feature_names = vectorizer.get_feature_names_out()


##
from random import sample
import re

sample_idx = 2
num_kwd = 3

sample_txt = corpus[sample_idx]
prep_sample = processed_corpus[sample_idx]

keyword = get_tfidf_keywords(vectorizer, feature_names, prep_sample, num_kwd)

yake_kwargs = {'lan': "en",
               "n": 1,
               "dedupLim": 0.9,
               "dedupFunc": 'seqm',
               "windowsSize": 1,
               }

import yake
keyword = get_YAKE_keywords(prep_sample, topk=6, **yake_kwargs)

kw_extractor = yake.KeywordExtractor(**yake_kwargs, top=20)
keywords = kw_extractor.extract_keywords(prep_sample)  # dict of {phrase: score}
keywords

# map keeyword indices to the original text
keyword_indices = []
for k in keyword.values():
    # print(sample_txt)
    # print(k)
    keyword_replaced = re.sub(k + r'\b', "[KWD]", sample_txt, flags=re.IGNORECASE)
    index = [idx for idx, w in enumerate(keyword_replaced.split(" ")) if w =="[KWD]"]
    keyword_indices.extend(index)


# select mask indices based on some rule
min_num_masks = int(num_kwd * 1)
random_seed = sum(keyword.keys())
banned_indices = [0,1]
remaining_mask_idx = [i for i in range(len(sample_txt.split(" "))) if i not in keyword_indices+banned_indices]
selected_mask_idx = sample(remaining_mask_idx, min_num_masks)

context = sample_txt.split(" ")
colored_context = sample_txt.split(" ")

for m_idx in selected_mask_idx:
    v = context[m_idx]
    context[m_idx] = "_"
    colored_context[m_idx] = f"\033[94m{v}\033[0m"

colored_context = " ".join(colored_context)
context = " ".join(context)
print(colored_context)
##

MODEL_DIR = './tmp/ilm/models/sto_ilm'
MASK_CLS = 'ilm.mask.hierarchical.MaskHierarchical'
import os
import pickle

import ilm.tokenize_util

tokenizer = ilm.tokenize_util.Tokenizer.GPT2
with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
    additional_ids_to_tokens = pickle.load(f)
additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
try:
    ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
except ValueError:
    print('Already updated')

# Load model
import torch
from transformers import GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()
_ = model.to(device)


# Create context
if selected_mask_idx[0] == 0:
    context = " " + context
context_ids = ilm.tokenize_util.encode(context, tokenizer)

# Replace blanks with appropriate tokens from left to right
_blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]
fill_idx = [idx for idx, _id in enumerate(context_ids) if _id == _blank_id]
for fidx in fill_idx:
    context_ids[fidx] = additional_tokens_to_ids['<|infill_word|>']





##

num_generate = 10
generated, g_idx, terminated, g_spans = infill_with_ilm(
    model,
    additional_tokens_to_ids,
    context_ids,
    num_infills=num_generate)

##
print(colored_context)
print("-"*50)
print("Original keywords:")
print(keyword)
original_keys = set(keyword.values())
for g, idx in zip(generated, g_idx):
    print('-' * 80)
    text = ilm.tokenize_util.decode(g, tokenizer)
    colored_text = ilm.tokenize_util.decode_with_color(g, tokenizer, color_idx=idx)
    print(colored_text)
    new_keyword = get_tfidf_keywords(vectorizer, feature_names, clean_text(text), num_kwd)
    new_keys = set(new_keyword.values())
    kwd_match_flag = original_keys == new_keys
    if not kwd_match_flag:
        print(color_text(original_keys == new_keys, colorcode="HEADER"))
        print(new_keyword)





##

