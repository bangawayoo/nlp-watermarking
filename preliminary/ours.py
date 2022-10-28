##
import math
from collections import defaultdict

import yake
from nltk import sent_tokenize
from nltk.corpus import stopwords
from itertools import product
import spacy

from utils import *

sr_threshold = 0.95
dtype = "imdb"
start_sample_idx = 0
num_sample = 30
stop = set(stopwords.words('english'))

# load dataset
corpus = load_dataset("imdb")['train']['text']
result_dir = f"results/ours-{dtype}-{start_sample_idx}-{start_sample_idx+num_sample}.txt"

cover_texts = corpus[start_sample_idx:start_sample_idx+num_sample]
cover_texts = [t.replace("\n", " ") for t in cover_texts]
cover_texts = preprocess_imdb(cover_texts)
nlp = spacy.load("en_core_web_sm")
match = total = 0
num_bits = 0
num_words = 0

for c_idx, cover_text in enumerate(cover_texts):
    sentences = sent_tokenize(cover_text)
    for s_idx, sen in enumerate(sentences):

        # c_idx, s_idx = 0, 1
        # sentences = sent_tokenize(cover_texts[c_idx])
        # sen = sentences[s_idx]

        print(f"{c_idx} {s_idx}")
        print(sen)
        punct_removed = sen.translate(str.maketrans(dict.fromkeys(string.punctuation)))
        word_count = len([i for i in punct_removed.split(" ") if i not in stop])
        num_words += word_count
        num_kwd = max(1, int(0.15 * word_count))
        # pick keywords
        yake_kwargs = {'lan': "en",
                       "n": 1,
                       "dedupLim": 0.9,
                       "dedupFunc": 'seqm',
                       "windowsSize": 1}

        # keyword = get_YAKE_keywords(cover_text, topk=6, **yake_kwargs)
        kw_extractor = yake.KeywordExtractor(**yake_kwargs, top=20)
        # truncate text to fir the max length limit (minus 10 to be safe)
        sen = tokenizer.decode(tokenizer(sen, add_special_tokens=False, truncation=True,
                        max_length=tokenizer.model_max_length// 2 - 10)['input_ids'])
        keywords = kw_extractor.extract_keywords(sen)  # dict of {phrase: score}
        selected_keywords = [k[0] for k in keywords[:num_kwd]]
        kw_hash = set(selected_keywords)

        # find dependencies related to the keyword (Mask candidates)

        kwd_dep = defaultdict(list)
        mask_dep = defaultdict(list)

        doc = nlp(sen)
        tokens_list = [token.text for token in doc]
        masked = []
        sub_words = []

        for token in doc:
            if token.text in selected_keywords:
                masked.append(token.head)
                mask_candidates = generate_substitute_candidates(tokens_list, token.head.i, topk=4)
                sub_words.append([x['token_str'] for x in mask_candidates])

        if not any([len(x) for x in sub_words]):
            match += 1
            total += 1
            print("Skipped\n")
            continue

        # Out of the mask candidates, find out which one is the best for a given criterion
        # criterion:
        #   - # of candidates that survived NLI threshold
        #   - # of candidates that passed keyword restoration
        valid_sub_words = []
        for idx, sw in enumerate(product(*sub_words)):
            wm_list = tokens_list.copy()
            for m_idx, m in enumerate(masked):
                wm_list[m.i] = sw[m_idx]
            new_keywords = kw_extractor.extract_keywords(" ".join(wm_list))
            new_keywords = [k[0] for k in new_keywords[:num_kwd]]
            nk_hash = set(new_keywords)
            if nk_hash == kw_hash:
                match += 1
                valid_sub_words.append(sw)
            total += 1

        if len(valid_sub_words):
            num_bits += math.log2(len(valid_sub_words))
        else:
            print(kw_hash)
            continue
        # sanity check
        for sw in valid_sub_words:
            wm_list = tokens_list.copy()
            for m_idx, m in enumerate(masked):
                wm_list[m.i] = sw[m_idx]
            new_keywords = kw_extractor.extract_keywords(" ".join(wm_list))
            new_keywords = [k[0] for k in new_keywords[:num_kwd]]
            nk_hash = set(new_keywords)
            if nk_hash != kw_hash:
                print("mismatch")


        print(kw_hash)
        print(nk_hash)
        print(masked)
        print(valid_sub_words)
        print("\n")

print(f"keyword match rate {match/total}")
print(f"bpw: {num_bits / num_words}")

# Choose rules




##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.chdir(os.path.join(os.getcwd(), "./preliminary"))
from preliminary.dataset_utils import arxiv_cs_abstracts

corpus = arxiv_cs_abstracts("train", attrs=['title', 'abstract'])
corpus = [i.replace("\n", " ") for i in corpus]

# corpus = roc_stories("test")
# corpus = [i.replace("\n", " ") for i in corpus]

from sklearn.feature_extraction.text import TfidfVectorizer
from preliminary.utils import *

max_corpus_sample = len(corpus)

stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(lowercase=True, stop_words=stop_words)

processed_corpus = [clean_text(a) for a in corpus[:max_corpus_sample]]
vectorizer.fit_transform(processed_corpus)
feature_names = vectorizer.get_feature_names_out()


