import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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

start_sample_idx = 0
num_sample = generic_args.num_sample
spacy_tokenizer = spacy.load(generic_args.spacy_model)

augmented_data_path = "./data/imdb-augmented-full.txt"
clean_text = []
corrupted_text = []
with open(augmented_data_path, "r") as reader:
    for line in reader:
        line = line.split("[sep]")
        for idx in range(len(line)-1):
            clean_text.append(line[0])
            corrupted_text.append(line[idx+1])

# shuffle the instances with fixed seed so that the clean-corrupted pairs are maintained
random.Random(0).shuffle(clean_text)
random.Random(0).shuffle(corrupted_text)


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
from transformers import DataCollatorForTokenClassification

def collator_for_masking(feature):
    datacollator = DataCollatorForTokenClassification(tokenizer, padding=True,
                                                      max_length=tokenizer.model_max_length,
                                                      return_tensors="pt")
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


# train model
pt_dataset = feature.train_test_split(
    train_size=0.6,
    test_size=0.4,
    shuffle=False
)
eval_dataset = pt_dataset['test']
if DEBUG_MODE:
    eval_dataset = eval_dataset.train_test_split(
    train_size=0.8,
    test_size=0.2,
    shuffle=False)
    eval_dataset = eval_dataset['test']



train_bs = 64
train_dl = DataLoader(
    pt_dataset['train'],
    shuffle=False,
    batch_size=train_bs,
    collate_fn=collator_for_masking
)
eval_dl = DataLoader(
    eval_dataset,
    shuffle=False,
    batch_size=train_bs*2,
    collate_fn=collator_for_masking
)

# log data as texts
# cnt = 0
# for b_idx, (batch, corr_batch) in enumerate(train_dl):
#     for b, cb in zip(batch["input_ids"], corr_batch["input_ids"]):
#         logger.info(tokenizer.decode(b).replace("[PAD]", ""))
#         logger.info(tokenizer.decode(cb).replace("[PAD]", "") + "\n")
#         cnt += 1
#     if cnt > 100:
#         break
# exit()

from torch.optim import AdamW
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")

#load from checkpoint
# state_dict = torch.load("./ckpt/debug/0")["model"]
# model.load_state_dict(state_dict)

optimizer = AdamW(model.parameters(), lr=5e-5)
fixed_model = copy.deepcopy(model)
fixed_model.eval()

from accelerate import Accelerator

accelerator = Accelerator()
model, fixed_model, optimizer, train_dl, eval_dl = accelerator.prepare(
    model, fixed_model, optimizer, train_dl, eval_dl
)

from transformers import get_scheduler

num_train_epochs = 300
num_update_steps_per_epoch = len(train_dl)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)



import torch
import torch.nn.functional as F
import math

progress_bar = tqdm(range(num_training_steps))
kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")
eval_freq = 5000
log_freq = 100
kl_weight = 1.0
topk = 32
optimize_topk = True
use_logit_loss = False
mse_criterion = torch.nn.MSELoss()
logit_loss_w = 1.0


EVAL_INIT = False
if EVAL_INIT:
    # Evaluation pre-training
    model.eval()
    losses = {"mlm": [], "r_mlm": []}
    for step, (batch, corr_batch) in enumerate(eval_dl):
        with torch.no_grad():
            outputs = model(**batch)
            masked_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            target_dist = F.softmax(outputs.logits[masked_index], dim=-1)

            corr_outputs = model(**corr_batch)
            masked_index = (corr_batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            pred_dist = F.softmax(corr_outputs.logits[masked_index], dim=-1)

            if optimize_topk:
                topk_target_dist, topk_target_idx = torch.topk(target_dist, topk, dim=-1)
                topk_pred_dist = []
                for k_idx in range(topk):
                    single_pred = pred_dist.gather(1, topk_target_idx[:, [k_idx]])
                    topk_pred_dist.append(single_pred)

                topk_pred_dist = torch.cat(topk_pred_dist, dim=1)
                topk_pred_dist = topk_pred_dist / topk_pred_dist.sum(dim=-1, keepdim=True)
                topk_target_dist = topk_target_dist / topk_target_dist.sum(dim=-1, keepdim=True)
                kl_loss = kl_criterion(topk_pred_dist.log(), topk_target_dist)
            else:
                kl_loss = kl_criterion(pred_dist.log(), target_dist)


        bs = batch['labels'].shape[0]
        loss = outputs.loss
        losses['mlm'].append(accelerator.gather(loss.repeat(bs)))
        losses['r_mlm'].append(accelerator.gather(kl_loss.repeat(bs)))

    logger.debug("Batch of topk pre-training:")
    topk_token_idx = torch.topk(pred_dist, 5, dim=-1)[1]
    for tti in topk_token_idx:
        logger.debug(tokenizer.decode(tti))

    mlm_losses = torch.cat(losses['mlm'])
    mlm_losses = mlm_losses[: len(pt_dataset['test'])].mean()

    rmlm_losses = torch.cat(losses['r_mlm'])
    rmlm_losses = rmlm_losses[: len(pt_dataset['test'])].mean()

    logger.info(f">>> Initial Perplexity: {mlm_losses:.2f}  R_mlm_kl: {rmlm_losses:.4f}")

step = 0

ckpt_dir = f"./ckpt/{generic_args.exp_name}/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

for epoch in range(num_train_epochs):
    # Training
    model.train()
    tr_losses = {"mlm": [], "r_mlm": [], "ll":[]}

    for b_idx, (batch, corr_batch) in enumerate(train_dl):
        with torch.no_grad():
            outputs = fixed_model(**batch)
            masked_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

        corr_outputs = model(**corr_batch)
        corr_masked_index = (corr_batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        ppl_loss = outputs.loss

        # the target distribution is detached from graph
        target_dist = F.softmax(outputs.logits[masked_index], dim=-1)
        pred_dist = F.softmax(corr_outputs.logits[corr_masked_index], dim=-1)
        if target_dist.shape[0] != pred_dist.shape[0]:
            logger.info(
                f"Number of masked tokens different for {b_idx} : target {target_dist.shape[0]} , pred: {pred_dist.shape[0]}")
            breakpoint()

        if optimize_topk:
            topk_target_dist, topk_target_idx = torch.topk(target_dist, topk, dim=-1)
            topk_pred_dist = []
            for k_idx in range(topk):
                single_pred = pred_dist.gather(1, topk_target_idx[:, [k_idx]])
                topk_pred_dist.append(single_pred)

            topk_pred_dist = torch.cat(topk_pred_dist, dim=1)
            topk_pred_dist = topk_pred_dist / topk_pred_dist.sum(dim=-1, keepdim=True)
            topk_target_dist = topk_target_dist / topk_target_dist.sum(dim=-1, keepdim=True)
            kl_loss = kl_criterion(topk_pred_dist.log(), topk_target_dist)
        else:
            kl_loss = kl_criterion(pred_dist.log(), target_dist)

        logit_loss = 0
        if use_logit_loss:
            target_logit = outputs.logits[masked_index]
            pred_logit = corr_outputs.logits[corr_masked_index]
            logit_loss = mse_criterion(pred_logit, target_logit)


        loss = kl_loss + logit_loss * logit_loss_w
        # print(kl_loss)
        # print(torch.tensor([p.norm() for p in model.parameters()]).mean())
        # breakpoint()
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        step += 1

        bs = batch['labels'].shape[0]
        tr_losses['mlm'].append(accelerator.gather(ppl_loss.repeat(bs)))
        tr_losses['r_mlm'].append(accelerator.gather(kl_loss.repeat(bs)))
        tr_losses['ll'].append(accelerator.gather(logit_loss.repeat(bs)))

        if step % log_freq == 0:
            # log training metric
            mlm_losses = torch.cat(tr_losses['mlm'])
            mlm_losses = mlm_losses.mean()
            rmlm_losses = torch.cat(tr_losses['r_mlm'])
            rmlm_losses = rmlm_losses.mean()
            logit_losses = torch.cat(tr_losses['ll'])
            logit_losses = logit_losses.mean()
            logger.info(f">>>Train log at Epoch {epoch}, Step {step}/{num_training_steps}: "
                        f"mlm: {mlm_losses:.3f}  R_mlm_kl: {rmlm_losses:.3f} logit_loss: {logit_losses:.3f}")
            tr_losses = {"mlm": [], "r_mlm": []}

        if step % eval_freq == 0:
            # Evaluation
            model.eval()
            losses = {"mlm":[], "r_mlm":[]}
            for batch, corr_batch in eval_dl:
                with torch.no_grad():
                    outputs = fixed_model(**batch)
                    masked_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

                    corr_outputs = model(**corr_batch)
                    corr_masked_index = (corr_batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

                    target_dist = F.softmax(outputs.logits[masked_index], dim=-1)
                    pred_dist = F.softmax(corr_outputs.logits[corr_masked_index], dim=-1)
                    if optimize_topk:
                        topk_target_dist, topk_target_idx = torch.topk(target_dist, topk, dim=-1)
                        topk_pred_dist = []
                        for k_idx in range(topk):
                            single_pred = pred_dist.gather(1, topk_target_idx[:, [k_idx]])
                            topk_pred_dist.append(single_pred)

                        topk_pred_dist = torch.cat(topk_pred_dist, dim=1)
                        topk_pred_dist = topk_pred_dist / topk_pred_dist.sum(dim=-1, keepdim=True)
                        topk_target_dist = topk_target_dist / topk_target_dist.sum(dim=-1, keepdim=True)
                        kl_loss = kl_criterion(topk_pred_dist.log(), topk_target_dist)
                    else:
                        kl_loss = kl_criterion(pred_dist.log(), target_dist)


                bs = batch['labels'].shape[0]
                loss = outputs.loss
                losses['mlm'].append(accelerator.gather(loss.repeat(bs)))
                losses['r_mlm'].append(accelerator.gather(kl_loss.repeat(bs)))

            logger.debug(f"At Step {step}:")
            topk_token_idx = torch.topk(pred_dist, 5, dim=-1)[1]
            for tti in topk_token_idx:
                logger.debug(tokenizer.decode(tti))

            mlm_losses = torch.cat(losses['mlm'])
            mlm_losses = mlm_losses[: len(pt_dataset['test'])].mean()

            rmlm_losses = torch.cat(losses['r_mlm'])
            rmlm_losses = rmlm_losses[: len(pt_dataset['test'])].mean()

            logger.info(f">>>Eval at Epoch {epoch}, Step {step}/{num_training_steps}: "
                        f"Perplexity: {mlm_losses:.3f}  R_mlm_kl: {rmlm_losses:.3f}")
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            accelerator.save(
                {"model": unwrapped.state_dict()},
                os.path.join(ckpt_dir, f"{step}.pth")
            )
            # unwrapped.save_pretrained(f"./ckpt/{generic_args.exp_name}/{step}")

accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)
accelerator.save(
    {"model": unwrapped.state_dict()},
    os.path.join(ckpt_dir, "last.pth")
)
accelerator.save_state(os.path.join(ckpt_dir, "last"))