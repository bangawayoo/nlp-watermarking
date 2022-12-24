import copy
import collections

from accelerate import Accelerator
import numpy as np
import torch
from transformers import DataCollatorForTokenClassification

from utils.infill_config import INFILL_TOKENIZER

tokenizer = INFILL_TOKENIZER

def collator_for_masking(feature):
    datacollator = DataCollatorForTokenClassification(tokenizer, padding=True,
                                                      max_length=tokenizer.model_max_length,
                                                      return_tensors="pt")
    corr_feature = []
    wwm_probability = 0.05

    for feat in feature:
        word_ids = feat.pop("word_ids", None)
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
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
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


def tokenize_function(example):
    result = tokenizer(example['text'])
    if tokenizer.is_fast:
      result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

