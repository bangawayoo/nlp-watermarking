from enum import Enum
import os
import random
import re
import spacy
import numpy as np
import logging

RAW_DATA_DIR = "./data/raw_data"

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

def preprocess_imdb(corpus):
  CLEANR = re.compile('<.*?>')
  corpus = [re.sub(CLEANR, '', c) for c in corpus]
  corpus = [re.sub(r"\.+", ".",c) for c in corpus]
  return corpus


def preprocess2sentence(corpus, start_sample_idx, num_sample, population_size=100):
  lengths = []
  sentence_tokenized = []
  nlp = spacy.load("en_core_web_sm")
  corpus = corpus[:population_size]
  for doc in nlp.pipe(corpus):
    lengths.extend([len(sent.text) for sent in doc.sents])
    sentence_tokenized.append([sent.text for sent in doc.sents])

  l_threshold = np.quantile(lengths, 0.05)
  u_threshold = np.quantile(lengths, 0.95)

  filtered = []
  num_skipped = 0
  num_processed = 0
  for sample in sentence_tokenized[start_sample_idx: start_sample_idx+num_sample]:
    sentences = [sen for sen in sample if len(sen) < u_threshold and len(sen) > l_threshold]
    num_skipped += len(sample) - len(sentences)
    num_processed += len(sentences)
    filtered.append(sentences)

  logging.info(f"{num_processed} sentences processed, {num_skipped} sentences skipped")
  return filtered