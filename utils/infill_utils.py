import copy
import collections
import os
import pickle

import numpy as np
import spacy
import torch
from transformers import DataCollatorForTokenClassification

from utils.infill_config import INFILL_TOKENIZER


tokenizer = INFILL_TOKENIZER

def collator_for_masking_random(feature, masking_p):
    datacollator = DataCollatorForTokenClassification(tokenizer, padding=True,
                                                      max_length=tokenizer.model_max_length,
                                                      return_tensors="pt")
    corr_feature = []
    for feat in feature:
        word_ids = feat.pop("word_ids", None)
        _ = feat.pop("text", None)
        corr_feat = {}
        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, masking_p, (len(mapping),))
        input_ids = feat["input_ids"]
        corr_feat['attention_mask'] = feat.pop("corr_attention_mask", None)
        corr_input_ids = feat.pop("corr_input_ids", None)
        labels = input_ids.copy()
        new_labels = [-100] * len(labels)
        corr_labels = [-100] * len(corr_input_ids)

        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            token_id = labels[min(mapping[word_id]):max(mapping[word_id]) + 1]
            # build window of length p centering the masked word
            p = 3
            window_start = max(min(mapping[word_id]) - p, 0)
            window_end = max(mapping[word_id]) + p
            window = corr_input_ids[window_start:window_end + 1]

            corr_token_idx = np.where(np.isin(window, token_id))[0] + window_start
            if len(corr_token_idx) == len(token_id):
                for t_idx in corr_token_idx:
                    corr_labels[t_idx] = copy.deepcopy(corr_input_ids[t_idx])
                    corr_input_ids[t_idx] = tokenizer.mask_token_id
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
        feat['input_ids'] = input_ids
        feat['labels'] = new_labels
        corr_feat['input_ids'] = corr_input_ids
        corr_feat['labels'] = corr_labels
        corr_feature.append(corr_feat)

    return datacollator(feature), datacollator(corr_feature)


spacy_tokenizer = spacy.load('en_core_web_sm')

def collator_for_masking_ours(feature, mask_selector, keyword_module):
    datacollator = DataCollatorForTokenClassification(tokenizer, padding=True,
                                                      max_length=tokenizer.model_max_length,
                                                      return_tensors="pt")
    corr_feature = []

    for feat in feature:
        word_ids = feat.pop("word_ids", None)
        text = feat.pop("text", None)
        sen = spacy_tokenizer(text)
        keywords, ent_keywords = keyword_module.extract_keyword([sen])
        mask_idx, mask = mask_selector.return_mask(sen, keywords[0], ent_keywords[0])
        mask_char_idx = [m.idx for m in mask]

        word_indices = []
        tokenized = tokenizer(text)
        # save indices of the start of the word
        for mci in mask_char_idx:
            char2token = tokenized.char_to_word(mci)
            word_indices.append(char2token)

        corr_feat = {}
        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # sanity check
        # for w_idx in word_indices:
        #     for t_idx in mapping[w_idx]:
        #         print(tokenizer.decode(feat['input_ids'][t_idx]))

        input_ids = feat["input_ids"]
        corr_feat['attention_mask'] = feat.pop("corr_attention_mask", None)
        corr_input_ids = feat.pop("corr_input_ids", None)
        labels = input_ids.copy()
        new_labels = [-100] * len(labels)
        corr_labels = [-100] * len(corr_input_ids)

        # use mapping to find word -> token indices
        for word_id in word_indices:
            word_id = word_id
            if len(mapping[word_id]) > 0:
                token_id = labels[min(mapping[word_id]):max(mapping[word_id]) + 1]
                # build window of length p centering the masked word
                p = 3
                window_start = max(min(mapping[word_id]) - p, 0)
                window_end = max(mapping[word_id]) + p
                window = corr_input_ids[window_start:window_end + 1]

                corr_token_idx = np.where(np.isin(window, token_id))[0] + window_start
                if len(corr_token_idx) == len(token_id):
                    for t_idx in corr_token_idx:
                        corr_labels[t_idx] = copy.deepcopy(corr_input_ids[t_idx])
                        corr_input_ids[t_idx] = tokenizer.mask_token_id
                    for idx in mapping[word_id]:
                        new_labels[idx] = labels[idx]
                        input_ids[idx] = tokenizer.mask_token_id
        feat['input_ids'] = input_ids
        feat['labels'] = new_labels
        corr_feat['input_ids'] = corr_input_ids
        corr_feat['labels'] = corr_labels
        corr_feature.append(corr_feat)

    return datacollator(feature), datacollator(corr_feature)


def featurize_for_masking_ours(feature, mask_selector, keyword_module, save_dir):

    corr_feature = []

    for feat in feature:
        word_ids = feat.pop("word_ids", None)
        text = feat.pop("text", None)
        sen = spacy_tokenizer(text)
        keywords, ent_keywords = keyword_module.extract_keyword([sen])
        mask_idx, mask = mask_selector.return_mask(sen, keywords[0], ent_keywords[0])
        mask_char_idx = [m.idx for m in mask]

        word_indices = []
        tokenized = tokenizer(text)
        # save indices of the start of the word
        for mci in mask_char_idx:
            char2token = tokenized.char_to_word(mci)
            word_indices.append(char2token)

        corr_feat = {}
        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # sanity check
        # for w_idx in word_indices:
        #     for t_idx in mapping[w_idx]:
        #         print(tokenizer.decode(feat['input_ids'][t_idx]))

        input_ids = feat["input_ids"]
        corr_feat['attention_mask'] = feat.pop("corr_attention_mask", None)
        corr_input_ids = feat.pop("corr_input_ids", None)
        labels = input_ids.copy()
        new_labels = [-100] * len(labels)
        corr_labels = [-100] * len(corr_input_ids)

        # use mapping to find word -> token indices
        for word_id in word_indices:
            word_id = word_id
            if len(mapping[word_id]) > 0:
                token_id = labels[min(mapping[word_id]):max(mapping[word_id]) + 1]
                # build window of length p centering the masked word
                p = 3
                window_start = max(min(mapping[word_id]) - p, 0)
                window_end = max(mapping[word_id]) + p
                window = corr_input_ids[window_start:window_end + 1]

                corr_token_idx = np.where(np.isin(window, token_id))[0] + window_start
                if len(corr_token_idx) == len(token_id):
                    for t_idx in corr_token_idx:
                        corr_labels[t_idx] = copy.deepcopy(corr_input_ids[t_idx])
                        corr_input_ids[t_idx] = tokenizer.mask_token_id
                    for idx in mapping[word_id]:
                        new_labels[idx] = labels[idx]
                        input_ids[idx] = tokenizer.mask_token_id
        feat['input_ids'] = input_ids
        feat['labels'] = new_labels
        corr_feat['input_ids'] = corr_input_ids
        corr_feat['labels'] = corr_labels
        corr_feature.append(corr_feat)

    with open(save_dir, "wb") as f:
        pickle.dump([feature, corr_feature], f)


def featurize_for_masking_random(feature, masking_p, save_dir):

    corr_feature = []

    for feat in feature:
        word_ids = feat.pop("word_ids", None)
        _ = feat.pop("text", None)
        corr_feat = {}
        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, masking_p, (len(mapping),))
        input_ids = feat["input_ids"]
        corr_feat['attention_mask'] = feat.pop("corr_attention_mask", None)
        corr_input_ids = feat.pop("corr_input_ids", None)
        labels = input_ids.copy()
        new_labels = [-100] * len(labels)
        corr_labels = [-100] * len(corr_input_ids)

        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            token_id = labels[min(mapping[word_id]):max(mapping[word_id]) + 1]
            # build window of length p centering the masked word
            p = 3
            window_start = max(min(mapping[word_id]) - p, 0)
            window_end = max(mapping[word_id]) + p
            window = corr_input_ids[window_start:window_end + 1]

            corr_token_idx = np.where(np.isin(window, token_id))[0] + window_start
            if len(corr_token_idx) == len(token_id):
                for t_idx in corr_token_idx:
                    corr_labels[t_idx] = copy.deepcopy(corr_input_ids[t_idx])
                    corr_input_ids[t_idx] = tokenizer.mask_token_id
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
        feat['input_ids'] = input_ids
        feat['labels'] = new_labels
        corr_feat['input_ids'] = corr_input_ids
        corr_feat['labels'] = corr_labels
        corr_feature.append(corr_feat)

    with open(save_dir, "wb") as f:
        pickle.dump([feature, corr_feature], f)

def collator_for_loading_pkl(path):
    datacollator = DataCollatorForTokenClassification(tokenizer, padding=True,
                                                      max_length=tokenizer.model_max_length,
                                                      return_tensors="pt")
    with open(path[0], "rb") as f:
        feature, corr_feature = pickle.load(f)

    return datacollator(feature), datacollator(corr_feature)


def tokenize_function(example):
    result = tokenizer(example['text'])
    if tokenizer.is_fast:
      result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

