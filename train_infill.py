import copy
import os
import collections
import random

from datasets import Dataset
import numpy as np
import spacy
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

from config import InfillArgs, GenericArgs, stop
from utils.logging import getLogger

random.seed(1230)

infill_parser = InfillArgs()
generic_parser = GenericArgs()
infill_args, _ = infill_parser.parse_known_args()
generic_args, _ = generic_parser.parse_known_args()
DEBUG_MODE = generic_args.debug_mode
dtype = generic_args.dtype

dirname = f'./logs/train-infill/{generic_args.exp_name}'
logger = getLogger("TRAIN-INFILL",
                   dir_=dirname,
                   debug_mode=DEBUG_MODE)

dirname = f"./results/ours/{dtype}/{generic_args.exp_name}"
start_sample_idx = 0
num_sample = generic_args.num_sample

spacy_tokenizer = spacy.load(generic_args.spacy_model)
if "trf" in generic_args.spacy_model:
    spacy.require_gpu()
# model = InfillModel(infill_args, dirname=dirname)

augmented_data_path = "./data/imdb-augmented.txt"
clean_text = []
corrupted_text = []
with open(augmented_data_path, "r") as reader:
    for line in reader:
        line = line.split("[sep]")
        clean_text.append(line[0])
        corrupted_text.append(line[1])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
batch = clean_text
corr_batch = corrupted_text

clean_dataset = Dataset.from_dict({"text": batch})
corr_dataset = Dataset.from_dict({"text": corr_batch})

def tokenize_function(example):
    result = tokenizer(example['text'])
    if tokenizer.is_fast:
      result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

feature = clean_dataset.map(tokenize_function, batched=True)
corr_feature = corr_dataset.map(tokenize_function, batched=True)

feature = feature.add_column("corr_input_ids", corr_feature['input_ids'])
feature = feature.add_column("corr_attention_mask", corr_feature['attention_mask'])
feature = feature.remove_columns("text")

from tqdm.auto import tqdm


# progress_bar = tqdm(range(len(feature['input_ids'])))

from transformers import DataCollatorForTokenClassification



def collator_for_masking(feature):
    datacollator = DataCollatorForTokenClassification(tokenizer, padding=True,
                                                      max_length=tokenizer.model_max_length,
                                                      return_tensors="pt")
    # breakpoint()
    wwm_probability = 0.15
    corr_feature = []

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
        # print(corr_input_ids)
        labels = input_ids.copy()
        new_labels = [0] * len(labels)
        corr_labels = [0] * len(corr_input_ids)

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


    # feature = feature.remove_columns(['input_ids', 'text', 'word_ids', 'corr_input_ids'])
    # feature = feature.add_column("input_ids", new_input_ids)
    # feature = feature.add_column("corr_input_ids", new_corr_input_ids)
    # feature = feature.add_column("labels", feature_labels)
    # feature = feature.add_column("corr_labels", corr_feature_labels)
    return datacollator(feature), datacollator(corr_feature)



# group text preprocessing
# chunk_size = 256
# def group_texts(examples):
#     # Concatenate all texts
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     # Compute length of concatenated texts
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the last chunk if it's smaller than chunk_size
#     total_length = (total_length // chunk_size) * chunk_size
#     # Split by chunks of max_len
#     result = {
#         k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
#         for k, t in concatenated_examples.items()
#     }
#     return result



# train model
pt_dataset = feature.train_test_split(
    train_size=0.6,
    test_size=0.4,
    shuffle=False
)

model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")



train_bs = 64
train_dl = DataLoader(
    pt_dataset['train'],
    shuffle=False,
    batch_size=train_bs,
    collate_fn=collator_for_masking
)
eval_dl = DataLoader(
    pt_dataset['test'],
    shuffle=False,
    batch_size=train_bs,
    collate_fn=collator_for_masking
)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dl, eval_dl = accelerator.prepare(
    model, optimizer, train_dl, eval_dl
)

from transformers import get_scheduler

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dl)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps,
)



import torch
import torch.nn.functional as F
import math

progress_bar = tqdm(range(num_training_steps))
kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")
save_every_ep = 1
kl_weight = 1.0


for epoch in range(num_train_epochs):
    # Training
    model.train()
    for b_idx, (batch, corr_batch) in enumerate(train_dl):
        outputs = model(**batch)
        loss = outputs.loss
        masked_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        # the target distribution is detached from graph
        target_dist = F.softmax(outputs.logits[masked_index], dim=-1).detach()

        corr_outputs = model(**corr_batch)
        masked_index = (corr_batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        pred_dist = F.softmax(corr_outputs.logits[masked_index], dim=-1)
        if target_dist.shape[0] != pred_dist.shape[0]:
            logger.info(f"Number of masked tokens different for {b_idx} : target {target_dist.shape[0]} , pred: {pred_dist.shape[0]}")
            breakpoint()
        kl_loss = kl_criterion(pred_dist.log(), target_dist)
        loss += kl_loss * kl_weight
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = {"mlm":[], "r_mlm":[]}
    for step, (batch, corr_batch) in enumerate(eval_dl):
        with torch.no_grad():
            outputs = model(**batch)
            masked_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            target_dist = F.softmax(outputs.logits[masked_index], dim=-1)

            corr_outputs = model(**corr_batch)
            masked_index = (corr_batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            pred_dist = F.softmax(corr_outputs.logits[masked_index], dim=-1)

            kl_loss = kl_criterion(pred_dist.log(), target_dist)

        bs = batch['labels'].shape[0]
        loss = outputs.loss
        losses['mlm'].append(accelerator.gather(loss.repeat(bs)))
        losses['r_mlm'].append(accelerator.gather(kl_loss.repeat(bs)))

    mlm_losses = torch.cat(losses['mlm'])
    mlm_losses = mlm_losses[: len(pt_dataset['test'])]
    try:
        perplexity = math.exp(torch.mean(mlm_losses))
    except OverflowError:
        perplexity = float("inf")

    rmlm_losses = torch.cat(losses['r_mlm'])
    rmlm_losses = rmlm_losses[: len(pt_dataset['test'])].mean()

    logger.info(f">>> Epoch {epoch}: Perplexity: {perplexity}  R_mlm_kl: {rmlm_losses}")

    if (epoch+1) % save_every_ep == 0:
        model.save_pretrained(f"./low_data-ckpt/{epoch}")