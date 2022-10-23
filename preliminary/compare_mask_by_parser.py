##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys
sys.path.append("/workspace/ilm/")
import pickle
from ilm.datasets import arxiv_cs_abstracts, roc_stories

NUM_SAMPLE = 100
corpus = arxiv_cs_abstracts("train", attrs=['title', 'abstract'])
# corpus = corpus[:1]
with open("./abs_parsed.pkl", 'rb') as f:
    parsed_results = pickle.load(f)

parsed_results = parsed_results[:NUM_SAMPLE]
##

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", is_split_into_words=True)
pipe_fill_mask = pipeline('fill-mask', model='bert-base-cased', top_k=32, device=0)
pipe_classification = pipeline(model="roberta-large-mnli", return_all_scores=True, device=0)


nltk.download('omw-1.4')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

stop = set(stopwords.words('english'))
puncuation = set(string.punctuation)
riskset = stop.union(puncuation)

##
def concatenate_for_ls(text, mask_idx):
    masked_text = text.copy()
    masked_text[mask_idx] = tokenizer.special_tokens_map['mask_token']
    tokens_for_ls = text + [tokenizer.special_tokens_map['sep_token']] + masked_text
    return " ".join(tokens_for_ls)


def generate_substitute_candidates(text, mask_idx, topk=5):
    # text = tokens
    # mask_idx = idx
    text_for_ls = concatenate_for_ls(text, mask_idx)
    tokenized = tokenizer(text_for_ls, truncation=False)
    if len(tokenized.input_ids) > 512:
        return []
    mask_candidates = pipe_fill_mask(text_for_ls)

    # filter out riskset
    filtered_mask_candidates = list(filter(lambda x: x['token_str'] not in riskset, mask_candidates))
    # filter out original word
    original_word = text[mask_idx]
    filtered_mask_candidates = list(filter(lambda x: x['token_str'].upper() != original_word.upper(), filtered_mask_candidates))

    # filter out morphorlogical derivations
    text_lm = lemmatizer.lemmatize(original_word, 'v')
    filtered_mask_candidates = list(
        filter(lambda x: lemmatizer.lemmatize(x['token_str'], 'v') != text_lm, filtered_mask_candidates)
    )

    for item in filtered_mask_candidates:
        masked_index = text_for_ls.index('[MASK]')
        seq_for_entail = text_for_ls.replace('[MASK]', item['token_str'])
        entail_result = pipe_classification(seq_for_entail)
        replaced = seq_for_entail[masked_index:]
        item['replaced_sequence'] = replaced
        item['entail_score'] = entail_result[0][2]['score']

    # sort in descending order
    filtered_mask_candidates = sorted(filtered_mask_candidates, key=lambda x: x['entail_score'], reverse=True)

    # filter out with sr_threshold
    filtered_mask_candidates = list(filter(lambda x: x['entail_score'] > sr_threshold, filtered_mask_candidates))
    return filtered_mask_candidates[:topk]  # top-2 words


##

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "gpt2-large"
lm = GPT2LMHeadModel.from_pretrained(model_id).to(device)
lm_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
##
import torch

def compute_ppl(texts):
    encodings = lm_tokenizer(texts, return_tensors='pt', padding='longest')
    encodings = {k: v.to(device) for k,v in encodings.items()}
    input_ids = encodings['input_ids']
    with torch.no_grad():
        output = lm(**encodings, labels=encodings['input_ids']).loss

    return torch.exp(output / input_ids.size(1)).item()



##
if __name__ == "__main__":
    sr_threshold=0.95
    lm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    lm_tokenizer.pad_token_id = 50256


    result = []
    for s_idx, sample in enumerate(parsed_results):
        tokens = list(sample[1])# [idx, tokens, tags, arcs]
        tags = sample[2]
        print(f"{s_idx}")
        print(*zip(sample[1], sample[2]), end="\n")

        replaced_ppls = []
        replaced_nums = []
        replaced_words = []

        for idx,tok in enumerate(tokens):
            if tags[idx] == "punct":
                replaced_ppls.append("NA")
                replaced_nums.append("NA")
                replaced_words.append("NA")
                continue
            filled = generate_substitute_candidates(tokens, idx)
            replaced_nums.append(len(filled))
            if len(filled):
                encodings = lm_tokenizer(" ".join(tokens), return_tensors='pt')
                text = [" ".join(tokens)]
                original_ppl = compute_ppl(text)
                replaced_texts = [x['replaced_sequence'] for x in filled]
                replaced_ppl = compute_ppl(replaced_texts)
                replaced_ppls.append(replaced_ppl - original_ppl)
                replaced_words.append([x['token_str'] for x in filled])
            else:
                replaced_ppls.append("NA")
                replaced_words.append("NA")
        result.append([replaced_ppls, replaced_words, replaced_nums])

    with open(f"./watermarked_results-{NUM_SAMPLE}.pkl", "wb") as f:
        pickle.dump(result, f)

##
# Edited for infill models to select words deterministically
def infill_with_ilm(
        model,
        special_tokens_to_ids,
        x,
        num_infills=1,
        max_sequence_length=256,
        nucleus=0.95, deterministic=False):
    _sep_id = special_tokens_to_ids['<|startofinfill|>']
    _end_span_id = special_tokens_to_ids['<|endofinfill|>']
    _special_ids = special_tokens_to_ids.values()

    # Make sure example doesn't already ends with [sep]
    if x[-1] == _sep_id:
        x = x[:-1]

    # Count number of blanks
    blank_idxs = []
    for i, tok_id in enumerate(x):
        if tok_id in _special_ids:
            blank_idxs.append(i)
    k = len(blank_idxs)
    if k == 0:
        raise ValueError()

    # Decode until we have that many blanks
    with torch.no_grad():
        device = next(model.parameters()).device
        context = torch.tensor(x + [_sep_id], dtype=torch.long, device=device).unsqueeze(0).repeat(num_infills, 1)

        terminated = []

        while context.shape[0] > 0:
            logits = model(context)[0][:, -1]
            if deterministic:
                next_tokens = select_from_logits(logits, nucleus=1, topk=5)
                next_tokens = torch.cat((next_tokens, torch.full_like(next_tokens, _end_span_id,
                                                                      device=next_tokens.device)), dim=-1)
                terminated.extend([list(s) for s in next_tokens.cpu().numpy()])
                break

            next_tokens = sample_from_logits(logits, nucleus=nucleus)
            context = torch.cat((context, next_tokens), dim=1)
            num_predicted_spans = (context == _end_span_id).long().sum(dim=1)


            terminate_expected = num_predicted_spans >= k
            terminate_toolong = torch.ones_like(context).long().sum(dim=1) >= max_sequence_length
            terminate = terminate_expected | terminate_toolong

            if torch.any(terminate):
                terminated_seqs = context[terminate, len(x) + 1:]
                terminated.extend([list(s) for s in terminated_seqs.cpu().numpy()])
                context = context[~terminate, :]

    # Collect generated spans
    generated_spans = []
    for gen in terminated:
        spans = []
        while _end_span_id in gen:
            spans.append(gen[:gen.index(_end_span_id)])
            gen = gen[gen.index(_end_span_id) + 1:]
        while len(spans) < k:
            spans.append([])
        generated_spans.append(spans)

    # Insert into context
    generated = []
    generated_indices = []
    for spans in generated_spans:
        context = copy.deepcopy(x)
        indices = []
        offset = 0
        for i, j in enumerate(blank_idxs[::-1]):
            del context[j]
            context[j:j] = spans[k - 1 - i]
        for i, j in enumerate(blank_idxs):
            indices.extend(range(j + offset, j + len(spans[i]) + offset))
            offset += len(spans[i]) - 1
        generated.append(context)
        generated_indices.append(indices)

    return generated, generated_indices, terminated, generated_spans


def select_from_logits(
        logits,
        temp=1.,
        topk=None,
        nucleus=1.):
    if temp == 0:
        return torch.argmax(logits, dim=-1).unsqueeze(-1)
    elif temp != 1:
        logits /= temp

    probs = F.softmax(logits, dim=-1)

    if topk is not None:
        top_probs = torch.topk(probs, topk)
        mask = F.one_hot(top_probs.indices, probs.shape[-1]).float()
        mask = mask.sum(dim=1)
        probs *= mask
        probs /= probs.sum(dim=-1)

    if nucleus != 1:
        probs_sorted = torch.sort(probs, descending=True, dim=-1)
        sorted_indices = probs_sorted.indices
        sorted_values = probs_sorted.values

        cumsum = torch.cumsum(sorted_values, dim=-1)
        ks = (cumsum < nucleus).long().sum(dim=-1)
        ks = torch.max(ks, torch.ones_like(ks))

        # TODO: Make this more efficient using gather
        ks = F.one_hot(ks, probs.shape[-1]).float()
        cutoffs = (sorted_values * ks).sum(-1)

        mask = (probs > cutoffs.unsqueeze(1)).float()
        probs *= mask

        probs /= probs.sum(keepdim=True, dim=-1)

    next_tokens = torch.nonzero(probs)[:,1].view(-1, 1)

    return next_tokens


