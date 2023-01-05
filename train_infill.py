import copy
from functools import partial
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random

from accelerate import Accelerator
from datasets import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler

from config import GenericArgs, InfillArgs, WatermarkArgs
from models.mask import MaskSelector
from models.kwd import KeywordExtractor
from utils.infill_config import INFILL_TOKENIZER, INFILL_MODEL
from utils.infill_utils import collator_for_masking_random, collator_for_masking_ours, tokenize_function
from utils.logging import getLogger


random.seed(1230)

# @record
def main():
    infill_parser = InfillArgs()
    generic_parser = GenericArgs()
    wm_parser = WatermarkArgs()
    infill_args, _ = infill_parser.parse_known_args()
    generic_args, _ = generic_parser.parse_known_args()
    wm_args, _ = wm_parser.parse_known_args()
    DEBUG_MODE = generic_args.debug_mode
    dtype = generic_args.dtype

    dirname = f'./logs/train-infill/{dtype}/{generic_args.exp_name}'
    logger = getLogger("TRAIN-INFILL",
                       dir_=dirname,
                       debug_mode=DEBUG_MODE)

    augmented_data_path = f"./data/{dtype}-augmented.txt"
    clean_text = []
    corrupted_text = []

    with open(augmented_data_path, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.split("[sep]")
            for idx in range(len(line)-1):
                clean_text.append(line[0])
                corrupted_text.append(line[idx+1])

    # shuffle the instances with a fixed seed so that the clean-corrupted pairs are maintained
    random.Random(0).shuffle(clean_text)
    random.Random(0).shuffle(corrupted_text)

    tokenizer = INFILL_TOKENIZER

    batch = clean_text
    corr_batch = corrupted_text

    clean_dataset = Dataset.from_dict({"text": batch})
    corr_dataset = Dataset.from_dict({"text": corr_batch})

    feature = clean_dataset.map(tokenize_function, batched=True)
    corr_feature = corr_dataset.map(tokenize_function, batched=True)

    feature = feature.add_column("corr_input_ids", corr_feature['input_ids'])
    feature = feature.add_column("corr_attention_mask", corr_feature['attention_mask'])

    mask_kwargs = {'method': wm_args.mask_select_method,
                   "mask_order_by": wm_args.mask_order_by,
                   "keyword_mask": wm_args.keyword_mask,
                   'exclude_cc': wm_args.exclude_cc}
    mask_selector = MaskSelector(**mask_kwargs)
    keyword_module = KeywordExtractor(ratio=wm_args.keyword_ratio)

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

    train_bs = 64 if not DEBUG_MODE else 8

    if infill_args.masking_type == "random":
        masking_p = infill_args.masking_p
        collate_func = partial(collator_for_masking_random, masking_p=masking_p)
    else:
        collate_func = partial(collator_for_masking_ours, mask_selector=mask_selector, keyword_module=keyword_module)

    # train_dataset = pt_dataset['train'].select(range(1))
    train_dataset = pt_dataset['train']
    train_dl = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=train_bs,
        collate_fn=collate_func
    )
    eval_dl = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=train_bs*2,
        collate_fn=collate_func
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

    model = INFILL_MODEL
    params = [p for n, p in model.named_parameters()]
    optimizer = AdamW(params, lr=5e-5)
    fixed_model = copy.deepcopy(model)
    fixed_model.eval()

    num_train_epochs = infill_args.num_epochs
    num_update_steps_per_epoch = len(train_dl)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0.1,
        num_training_steps=num_training_steps,
    )

    accelerator = Accelerator()
    # load from checkpoint
    if infill_args.model_ckpt:
        model.from_pretrained(infill_args.model_ckpt)
        optim_scheduler_states = torch.load(os.path.join(infill_args.model_ckpt, "/optim_state.pth"))

        logger.info("Loading optimizer states from checkpoint dir ..")
        optimizer.load_state_dict(optim_scheduler_states["optimizer"])
        completed_epochs = optim_scheduler_states["epoch"]
        completed_steps = optim_scheduler_states["steps"]
        lr_scheduler.load_state_dict(optim_scheduler_states["scheduler"])

    model, fixed_model, optimizer, train_dl, eval_dl = accelerator.prepare(
        model, fixed_model, optimizer, train_dl, eval_dl
    )

    kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")
    eval_freq = 20000
    log_freq = 1000
    kl_weight = 1.0
    topk = 32
    optimize_topk = True
    use_logit_loss = False
    optimize_cls_token = False
    mse_criterion = torch.nn.MSELoss()
    logit_loss_w = 1.0
    kl_type = infill_args.kl_type

    ckpt_dir = f"./ckpt/{dtype}/{generic_args.exp_name}/"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    def compute_loss(target_dist, pred_dist, kl_criterion, target_logit, pred_logit,
                     mse_criterion=None, optimize_topk=False,
                     use_logit_loss=False, kl_type="forward"):
        # implement accuracy as metric
        _, topk_target_idx = torch.topk(target_dist, topk, dim=-1)
        _, topk_pred_idx = torch.topk(pred_dist, topk, dim=-1)
        acc_list = []

        for p_row, t_row in zip(topk_pred_idx, topk_target_idx):
            isin_mask = torch.isin(p_row, t_row)
            acc = isin_mask.sum() / isin_mask.numel()
            acc_list.append(acc.unsqueeze(-1))

        if optimize_topk:
            row_idx = [[idx] * topk_target_idx.shape[1] for idx in range(topk_target_idx.shape[0])]
            row_idx = [item for sublist in row_idx for item in sublist]
            col_idx = torch.flatten(topk_target_idx).tolist()

            bool_mask = torch.empty(target_dist.shape, dtype=torch.bool, device=target_dist.device)
            bool_mask[:] = True
            bool_mask[row_idx, col_idx] = False
            target_dist[bool_mask] = 0
            target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)
            # target_dist[row_idx, col_idx] = 1
            target_dist = target_dist + 1e-12

        if kl_type == "reverse":
            # use reverse kl
            kl_loss = kl_criterion(target_dist.log(), pred_dist)
        else:
            # forward kl
            kl_loss = kl_criterion(pred_dist.log(), target_dist)

            ### optimizing only for the topk ###
            # topk_target_dist, topk_target_idx = torch.topk(target_dist, topk, dim=-1)
            # topk_pred_dist = []
            # for k_idx in range(topk):
            #     single_pred = pred_dist.gather(1, topk_target_idx[:, [k_idx]])
            #     topk_pred_dist.append(single_pred)
            #
            # topk_pred_dist = torch.cat(topk_pred_dist, dim=1)
            # topk_pred_dist = topk_pred_dist / topk_pred_dist.sum(dim=-1, keepdim=True)
            # topk_target_dist = topk_target_dist / topk_target_dist.sum(dim=-1, keepdim=True)

            # kl_loss = kl_criterion(topk_pred_dist.log(), topk_target_dist)

        logit_loss = torch.tensor(-1, dtype=torch.float, device=target_dist.device)
        if use_logit_loss:
            logit_loss = mse_criterion(pred_logit, target_logit)

        if kl_loss == float("inf") or kl_loss == float("-inf"):
            logger.info("KL loss is inf!")
            breakpoint()

        if acc_list:
            acc_list = torch.cat(acc_list)

        return kl_loss, logit_loss, acc_list


    def evaluate(eval_dl, epoch, step, save_ckpt=False):
        model.eval()
        losses = {"mlm": [], "r_mlm": [], "acc": [], 'll': []}
        for batch, corr_batch in eval_dl:
            with torch.no_grad():
                outputs = fixed_model(**batch)
                masked_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

                corr_outputs = model(**corr_batch)
                corr_masked_index = (corr_batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

                target_dist = F.softmax(outputs.logits[masked_index], dim=-1)
                pred_dist = F.softmax(corr_outputs.logits[corr_masked_index], dim=-1)

                kl_loss, logit_loss, acc = compute_loss(target_dist, pred_dist, kl_criterion,
                                                        outputs.logits[masked_index],
                                                        corr_outputs.logits[corr_masked_index],
                                                        mse_criterion=mse_criterion,
                                                        optimize_topk=optimize_topk,
                                                        use_logit_loss=use_logit_loss,
                                                        kl_type=kl_type)

            bs = batch['labels'].shape[0]
            loss = corr_outputs.loss
            losses['mlm'].append(accelerator.gather(loss.repeat(bs)))
            losses['r_mlm'].append(accelerator.gather(kl_loss.repeat(bs)))
            if len(acc):
                losses['acc'].append(acc)
            losses['ll'].append(accelerator.gather(logit_loss.repeat(bs)))


        logger.debug(f"At Step {step}:")
        topk_token_idx = torch.topk(pred_dist, 5, dim=-1)[1]
        for tti in topk_token_idx:
            logger.debug(tokenizer.decode(tti))

        log_output = ""
        for k, v in losses.items():
            if len(v):
                mean_loss = torch.cat(v)[: len(pt_dataset['test'])].mean()
                log_output += f"{k}: {mean_loss:.3f}\t"
                losses[k] = []

        logger.info(f">>>Eval at Epoch {epoch}, Step {step}/{num_training_steps}\t"
                    f"{log_output}")

        if save_ckpt:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                os.path.join(ckpt_dir, f"{step}")
            )
            accelerator.save(
                {
                    "epoch": epoch,
                    "steps": step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                },
                os.path.join(ckpt_dir, f"{step}/optim_state.pth")
            )

    if infill_args.eval_init or infill_args.eval_only:
        logger.info("Evaluating...")
        # Evaluation pre-training
        evaluate(eval_dl, 0, 0, save_ckpt=False)
        if infill_args.eval_only:
            exit()

    step = 0
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Train metric
        tr_losses = {"mlm": [], "r_mlm": [], "acc": [], "ll": []}

        for b_idx, (batch, corr_batch) in enumerate(train_dl):
            model.train()
            with torch.no_grad():
                outputs = fixed_model(**batch)
                if optimize_cls_token:
                    masked_index = torch.logical_or(batch['input_ids'] == tokenizer.mask_token_id,
                                                batch['input_ids'] == 101).nonzero(as_tuple=True)
                else:
                    masked_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            corr_outputs = model(**corr_batch)
            if optimize_cls_token:
                corr_masked_index = torch.logical_or(corr_batch['input_ids'] == tokenizer.mask_token_id,
                                                 corr_batch['input_ids'] == 101).nonzero(as_tuple=True)
            else:
                corr_masked_index = (corr_batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            ppl_loss = corr_outputs.loss
            # the target distribution is detached from graph
            target_dist = F.softmax(outputs.logits[masked_index], dim=-1)
            pred_dist = F.softmax(corr_outputs.logits[corr_masked_index], dim=-1)

            if target_dist.shape[0] != pred_dist.shape[0]:
                logger.info(
                    f"Number of masked tokens different for {b_idx} : target {target_dist.shape[0]} , pred: {pred_dist.shape[0]}")
                breakpoint()

            kl_loss, logit_loss, acc = compute_loss(target_dist, pred_dist, kl_criterion,
                                                    outputs.logits[masked_index], corr_outputs.logits[corr_masked_index],
                                                    mse_criterion=mse_criterion,
                                                    optimize_topk=optimize_topk,
                                                    use_logit_loss=use_logit_loss,
                                                    kl_type=kl_type)
            if kl_loss == float("inf") or kl_loss == float("-inf"):
                logger.info("KL loss is inf!")
                breakpoint()
            loss = kl_loss + logit_loss * logit_loss_w
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step += 1

            bs = batch['labels'].shape[0]
            tr_losses['mlm'].append(accelerator.gather(ppl_loss.detach().repeat(bs)))
            tr_losses['r_mlm'].append(accelerator.gather(kl_loss.detach().repeat(bs)))
            if len(acc):
                tr_losses['acc'].append(acc)
            tr_losses['ll'].append(accelerator.gather(logit_loss.detach().repeat(bs)))

            if step % log_freq == 0:
                log_output = ""
                for k, v in tr_losses.items():
                    if len(v):
                        mean_loss = torch.cat(v).mean()
                        log_output += f"{k}: {mean_loss:.3f}\t"
                        tr_losses[k] = []
                logger.info(f">>>Train log at Epoch {epoch}, Step {step}/{num_training_steps}\t"
                            f"{log_output}")

            if step % eval_freq == 0 or step == num_training_steps:
                # Evaluation
                evaluate(eval_dl, epoch, step, save_ckpt=True)

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        os.path.join(ckpt_dir, f"last")
    )
    accelerator.save(
        {
            "epoch": epoch,
            "steps": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict()
        },
        os.path.join(ckpt_dir, "last/optim_state.pth")
    )

if __name__ == "__main__":
    main()