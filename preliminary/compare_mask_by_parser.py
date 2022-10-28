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



