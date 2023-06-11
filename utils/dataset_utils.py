from enum import Enum
import os
import pickle
import numpy as np
import random
import re

from datasets import load_dataset
import spacy
from tqdm import tqdm
from utils.logging import getLogger

RAW_DATA_DIR = "./data/raw_data"
CACHE_DIR= "./data/cache"
logger = getLogger("DATASET_UTILS", debug_mode=True)

class Dataset(Enum):
  CUSTOM = 0
  ARXIV_CS_ABSTRACTS = 1
  ROC_STORIES = 2
  ROC_STORIES_NO_TITLE = 3
  LYRICS_STANZAS = 4


def get_dataset(dataset, split, *args, data_dir=None, shuffle=False, limit=None, **kwargs):
  if type(dataset) != Dataset:
    raise ValueError('Must specify a Dataset enum value')

  if dataset == Dataset.CUSTOM:
    d = custom(split, data_dir)
    if data_dir is None:
      raise ValueError('Data dir must be specified for custom dataset')
  elif dataset == Dataset.ARXIV_CS_ABSTRACTS:
    d = arxiv_cs_abstracts(split, *args, data_dir=data_dir, **kwargs)
  elif dataset == Dataset.ROC_STORIES:
    d = roc_stories(split, *args, data_dir=data_dir, **kwargs)
  elif dataset == Dataset.ROC_STORIES_NO_TITLE:
    d = roc_stories(split, *args, data_dir=data_dir, with_titles=False, **kwargs)
  elif dataset == Dataset.LYRICS_STANZAS:
    assert split in ['train', 'valid', 'test']
    if data_dir is None:
      data_dir = os.path.join(RAW_DATA_DIR, 'lyrics_stanzas')
    d = custom(split, data_dir=data_dir)
  else:
    assert False

  if shuffle:
    random.shuffle(d)

  if limit is not None:
    d = d[:limit]

  return d


def custom(split, data_dir):
  fp = os.path.join(data_dir, '{}.txt'.format(split))
  try:
    with open(fp, 'r') as f:
      entries = [e.strip() for e in f.read().strip().split('\n\n\n')]
  except:
    raise ValueError('Could not load from {}'.format(fp))
  return entries


ABS_DIR = os.path.join(RAW_DATA_DIR, 'arxiv_cs_abstracts')
def arxiv_cs_abstracts(split='train', data_dir=None, attrs=['title', 'authors', 'categories', 'abstract']):
  assert split in ['train', 'valid', 'test']

  if data_dir is None:
    data_dir = ABS_DIR

  with open(os.path.join(data_dir, 'arxiv_cs_abstracts.txt'), 'r') as f:
    raw = f.read().split('\n\n\n')

  abstracts = []
  for r in raw:
    aid, created, updated, categories, title, authors, abstract = r.split('\n', 6)

    a = []
    for attr_name in attrs:
      a.append(eval(attr_name))
    a = '\n'.join(a)

    if created.startswith('2018'):
      if split == 'valid':
        abstracts.append(a)
    elif created.startswith('2019'):
      if split == 'test':
        abstracts.append(a)
    else:
      if split == 'train':
        abstracts.append(a)

  return abstracts


ROC_STORIES_DIR = os.path.join(RAW_DATA_DIR, 'roc_stories')
def roc_stories(split='train', data_dir=None, with_titles=True, exclude_nonstandard=True):
  assert split in ['train', 'valid', 'test', 'test_hand_title']

  if data_dir is None:
    data_dir = ROC_STORIES_DIR

  if split == 'train':
    with open(os.path.join(data_dir, 'train_title.txt'), 'r') as f:
      stories = f.read().split('\n\n\n')
    titled = True
  elif split == 'valid':
    with open(os.path.join(data_dir, 'valid.txt'), 'r') as f:
      stories = f.read().split('\n\n\n')
    titled = False
  elif split == 'test':
    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
      stories = f.read().split('\n\n\n')
    titled = False
  elif split == 'test_hand_title':
    with open(os.path.join(data_dir, 'test_hand_title.txt'), 'r') as f:
      stories = f.read().split('\n\n\n')
    titled = True

  stories = [s.strip() for s in stories if len(s.strip()) > 0]

  if with_titles != titled:
    if with_titles:
      stories = ['Unknown Title\n{}'.format(s) for s in stories]
    else:
      stories = [s.splitlines()[-1] for s in stories]

  if exclude_nonstandard:
    from nltk.tokenize import sent_tokenize

    standardized = []
    for s in stories:
      paragraphs = s.splitlines()
      if len(paragraphs) != (2 if with_titles else 1):
        continue
      try:
        if len(sent_tokenize(paragraphs[-1])) != 5:
          continue
      except:
        raise Exception('Need to call nltk.download(\'punkt\')')
      standardized.append(s)
    stories = standardized

  return stories

def preprocess_txt(corpus):
    corpus = [t.replace("\n", " ") for t in corpus]
    corpus = [t.replace("\t", " ") for t in corpus]
    # html tag
    CLEANR = re.compile('<.*?>')
    corpus = [re.sub(CLEANR, '', c) for c in corpus if len(c) > 0]
    corpus = [re.sub(r"\.+", ".", c) for c in corpus]
    return corpus


def change_str_to_int(listed_str):
    if len(listed_str) < 1:
      return
    int_str = [int(elem) for elem in listed_str if elem.isdigit()]
    return int_str


def get_result_txt(result_txt, sep='\t'):
    results = []
    with open(f"{result_txt}", "r") as reader:
        for line in reader:
            line = line.split(sep)
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
                line[4] = line[4].strip() if len(line[4]) > 0 else ""
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


def preprocess2sentence(corpus, corpus_name, start_sample_idx, num_sample=3000,
                        population_size=3000, cutoff_q=(0.05, 0.95),
                        spacy_model="en_core_web_sm", use_cache=True):
    population_size = max(population_size, start_sample_idx + num_sample)
    id = f"{corpus_name}-{spacy_model}"
    if corpus_name in ['dracula']:
        cutoff_q = (0.1, 0.9)
    if not os.path.isdir(CACHE_DIR):
      os.makedirs(CACHE_DIR, exist_ok=True)
    file_dir = os.path.join(CACHE_DIR, id+".pkl")

    if use_cache and os.path.isfile(file_dir):
      logger.info(f"Using cache {file_dir}")
      with open(file_dir, "rb") as f:
        docs = pickle.load(f)

    else:
      logger.info(f"Processing corpus with {spacy_model}...")
      nlp = spacy.load(spacy_model)
      corpus = corpus[:population_size]
      docs = []
      num_workers = 4
      if "trf" in spacy_model:
        for c in corpus:
          docs.append(nlp(c))
      else:
        for doc in nlp.pipe(corpus, n_process=num_workers):
          docs.append(doc)

      logger.info(f"Caching preprocessed sentences")
      with open(file_dir, "wb") as f:
        pickle.dump(docs, f)

    lengths = []
    sentence_tokenized = []
    sentence_list = []

    for doc in docs:
        lengths.extend([len(sent) for sent in doc.sents if len(sent.text.strip()) > 0])
        sentence_tokenized.append([sent for sent in doc.sents if len(sent.text.strip()) > 0])
        sentence_list.extend([(sent, len(sent)) for sent in doc.sents if len(sent.text.strip()) > 0])

    sentence_list.sort(key=lambda x: x[1])  # for debugging
    l_threshold = np.quantile(lengths, cutoff_q[0])
    # manually set upper threshold to 200 just to be safe when using Pretrained Tokenizers with maximum length=512.
    u_threshold = min(np.quantile(lengths, cutoff_q[1]), 200)

    filtered = []
    num_skipped = 0
    num_processed = 0

    for sample in sentence_tokenized[start_sample_idx: start_sample_idx+num_sample]:
        sentences = [sen for sen in sample if l_threshold <= len(sen) <= u_threshold]
        num_skipped += len(sample) - len(sentences)
        num_processed += len(sentences)
        filtered.append(sentences)

    # post_processed_length = []
    # for sentences in filtered:
    #     post_processed_length.extend([len(sen) for sen in sentences])
    #
    # print([np.quantile(post_processed_length, q) for q in [0.1, 0.25, 0.5, 0.75, 0.9]])
    # print(min(post_processed_length))
    # print(max(post_processed_length))
    logger.info(f"{num_processed} sentences processed, {num_skipped} sentences skipped")
    return filtered

def get_dataset(dtype):
    if dtype == "imdb":
        corpus = load_dataset(dtype)['train']['text']
        test_corpus = load_dataset(dtype)['test']['text']
        train_num_sample = len(corpus)
        test_num_sample = 2000
    elif dtype == "wikitext":
        corpus = load_dataset(dtype, 'wikitext-2-raw-v1')['train']['text']
        test_corpus = load_dataset(dtype, 'wikitext-2-raw-v1')['test']['text']
        corpus = [t for t in corpus if not t.strip().startswith("=") and len(t.strip())> 0]
        test_corpus = [t for t in test_corpus if not t.strip().startswith("=") and len(t.strip()) > 0]
        train_num_sample = len(corpus)
        test_num_sample = len(corpus)
    elif dtype == "agnews":
        corpus = load_dataset("ag_news")['train']['text']
        test_corpus = load_dataset("ag_news")['test']['text']
        train_num_sample = len(corpus)
        test_num_sample = len(test_corpus)
    elif dtype in ['dracula', 'wuthering_heights']:
        with open(f"./data/{dtype}.txt", "r") as f:
            text = f.readlines()
        text_string = " ".join(text).replace("\n", "")
        text_list = text_string.split("  ")
        split_point = int(0.4 * len(text_list))
        corpus = text_list[:split_point]
        test_corpus = text_list[split_point:]
        train_num_sample = len(corpus)
        test_num_sample = len(test_corpus)
    else:
        raise NotImplementedError

    return corpus, test_corpus, {'train': train_num_sample, 'test': test_num_sample}