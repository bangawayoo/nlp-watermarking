from datetime import datetime
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy
import sys
sys.path.append(os.getcwd())
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM


from dataset_utils import preprocess2sentence, preprocess_imdb
from utils import *


# TODO:
now = datetime.now()
dt_string = now.strftime("%m%d_%H%M%S")
stop = set(stopwords.words('english'))
punctuation = set(string.punctuation)
riskset = stop.union(punctuation)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")

log_path = f"./logs/infill/infill-{dt_string}.log"
log_dir = os.path.dirname(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(logFormatter)
logger.addHandler(file_handler)

class InfillModel:
    def __init__(self):
        self.device = torch.device('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.lm_head = AutoModelForMaskedLM.from_pretrained('bert-base-cased').to(self.device)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(self.device)
        self.nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.threshold = 0.1
        self.nli_threshold = 0.95
        self.word_tokenizer = spacy.load("en_core_web_sm")
        self.grad_cnt = 0
        params = [p for p in self.lm_head.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=5e-5)
        self.metric = {'entail_score':[], 'num_subs':[]}
        self.train_d, self.test_d = self._init_dataset()
        self.kwd_kwargs = {'lan': "en",
                           "n": 1,
                           "dedupLim": 0.9,
                           "dedupFunc": 'seqm',
                           "windowsSize": 1}
        self.train_kwargs = {'epoch':100,
                             }

    def fill_mask(self, text, mask_indices, train_flag=True, topk=2):
        # sample text = "By now you should know that the capital of France is Paris."
        tokenized_text = [token.text for token in self.word_tokenizer(text)]
        masked_tokenized = []
        for m_idx in mask_indices:
            tmp = tokenized_text.copy()
            tmp[m_idx] = self.tokenizer.special_tokens_map['mask_token']
            masked_tokenized.append(tmp)

        inputs = self.tokenizer(masked_tokenized, return_tensors="pt", add_special_tokens=True, is_split_into_words=True,
                                padding="longest")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        if train_flag:
            logits = self.lm_head(**inputs).logits
        else:
            with torch.no_grad():
                logits = self.lm_head(**inputs).logits

        masked_index = torch.nonzero(inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=True)
        # Fill mask pipeline supports only one mask_token per sample
        mask_logits = logits[masked_index]
        probs = mask_logits.softmax(dim=-1)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)

        candidate_texts = []
        valid_input = [True for _ in range(len(mask_indices))]
        avg_num_cand = 0
        # filter the chosen tokens
        for idx, m_idx in enumerate(mask_indices):
            candidate_ids = self._filter_words(indices[idx,:32].squeeze(), tokenized_text[m_idx]).to(self.device)
            avg_num_cand += len(candidate_ids)
            if len(candidate_ids) == 0:
                # candidate_ids = inputs['input_ids'][idx, m_idx].clone().repeat(topk)
                valid_input[idx] = False
                continue
            elif len(indices) < topk:
                candidate_ids = candidate_ids.repeat(topk)[:topk]
            else:
                candidate_ids = candidate_ids[:topk]

            compared_inputs = inputs['input_ids'][idx].clone().repeat(topk, 1) # (batch, seq)
            compared_inputs[:, m_idx] = candidate_ids
            candidate_texts.append(compared_inputs)
        avg_num_cand /= len(mask_indices)

        if len(candidate_texts) > 0 :
            candidate_texts = torch.cat(candidate_texts, dim=0)
            # run NLI with the original input
            nli_input = self._concatenate_for_nli(candidate_texts, text)
            nli_input = {k: v.to(self.device) for k, v in nli_input.items()}
            with torch.no_grad():
                nli_output = self.nli_model(**nli_input)
            entail_score = nli_output.logits.softmax(dim=-1)[:, 2]
            regret = ((0.95 - entail_score) * sorted_probs[valid_input, :topk].flatten()).sum()

            if train_flag:
                regret.backward()
            self.grad_cnt += len(entail_score)

            logger.info(f"Masked text: {[tokenized_text[m_idx] for m_idx in mask_indices]}")
            # logger.info(f"Substituted texts: {self.tokenizer.decode(chosen_word_idx)}")
            # logger.info(sum([p.grad.norm() for p in self.lm_head.parameters()]))
            # logger.info(entail_score.mean().item())
            self.metric['entail_score'].extend(entail_score.tolist())
            self.metric['num_subs'].append(avg_num_cand)

        if self.grad_cnt >= 128 and train_flag:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.grad_cnt = 0
            logger.info(f"entail score: {entail_score.mean().item()}")

    def train(self):
        for ep in range(self.train_kwargs['epoch']):
            logger.info(f"Epoch {ep}")
            for c_idx, sentences in enumerate(self.train_d):
                for s_idx, sen in enumerate(sentences):
                    logger.info(f"{c_idx} {s_idx}")
                    logger.info(sen)
                    # truncate text to the max length limit (minus 10 to be safe)
                    sen = tokenizer.decode(tokenizer(sen, add_special_tokens=False, truncation=True,
                                                     max_length=tokenizer.model_max_length // 2 - 10)['input_ids'])

                    doc = self.word_tokenizer(sen)
                    punct_removed = [token.text for token in doc if not token.is_punct]

                    word_count = len(punct_removed)
                    num_kwd = max(1, int(0.15 * word_count))
                    # pick keywords
                    kw_extractor = yake.KeywordExtractor(**self.kwd_kwargs, top=20)
                    # truncate text to the max length limit (minus 10 to be safe)
                    sen = tokenizer.decode(tokenizer(sen, add_special_tokens=False, truncation=True,
                                                     max_length=tokenizer.model_max_length // 2 - 10)['input_ids'])
                    keywords = kw_extractor.extract_keywords(sen)  # dict of {phrase: score}
                    selected_keywords = [k[0] for k in keywords[:num_kwd]]
                    kw_hash = set(selected_keywords)

                    doc = self.word_tokenizer(sen)
                    tokens_list = [token.text for token in doc]
                    masked = []

                    for token in doc:
                        if token.text in selected_keywords:
                            masked.append((token.head, token.head.i))
                            selected_keywords.remove(token.text)

                    if masked:
                        mask_indices = [i[1] for i in masked]
                        model.fill_mask(sen, mask_indices)

    def evaluate(self):
        for c_idx, sentences in enumerate(self.test_d):
            for s_idx, sen in enumerate(sentences):
                logger.info(f"{c_idx} {s_idx}")
                logger.info(sen)
                # truncate text to the max length limit (minus 10 to be safe)
                sen = tokenizer.decode(tokenizer(sen, add_special_tokens=False, truncation=True,
                                                 max_length=tokenizer.model_max_length // 2 - 10)['input_ids'])

                doc = self.word_tokenizer(sen)
                punct_removed = [token.text for token in doc if not token.is_punct]
                word_count = len(punct_removed)
                num_kwd = max(1, int(0.15 * word_count))

                # pick keywords
                kw_extractor = yake.KeywordExtractor(**self.kwd_kwargs, top=20)
                keywords = kw_extractor.extract_keywords(sen)  # dict of {phrase: score}
                selected_keywords = [k[0] for k in keywords[:num_kwd]]
                kw_hash = set(selected_keywords)
                masked = []

                for token in doc:
                    if token.text in selected_keywords:
                        masked.append((token.head, token.head.i))
                        selected_keywords.remove(token.text)

                if masked:
                    mask_indices = [i[1] for i in masked]
                    model.fill_mask(sen, mask_indices, train_flag=False)



    def init_metric(self):
        self.metric = {'entail_score': [], 'num_subs': []}

    def _init_dataset(self, dtype="imdb"):
        start_sample_idx = 0
        corpus = load_dataset("imdb")['train']['text']
        num_sample = len(corpus)

        cover_texts = [t.replace("\n", " ") for t in corpus]
        cover_texts = preprocess_imdb(cover_texts)
        cover_texts = preprocess2sentence(cover_texts, start_sample_idx, num_sample)

        corpus = load_dataset("imdb")['test']['text']
        test_cover_texts = [t.replace("\n", " ") for t in corpus]
        test_cover_texts = preprocess_imdb(test_cover_texts)
        test_cover_texts = preprocess2sentence(test_cover_texts, start_sample_idx, 1000)
        return cover_texts, test_cover_texts

    def _concatenate_for_nli(self, compared_ids, text):
        num_samples = compared_ids.shape[0]
        compared_text = self.tokenizer.batch_decode(compared_ids, skip_special_tokens=True)
        text = [text] * num_samples
        nli_input_encoded = self.nli_tokenizer(text, compared_text, add_special_tokens=True, return_tensors='pt',
                                               padding='longest')

        return nli_input_encoded

    def _filter_words(self, candidate_ids, original_word):
        mask_candidates = [{'token_str': self.tokenizer.decode(x),
                            'token': x}
                           for x in candidate_ids]
        mask_candidates = list(filter(lambda x: not (x['token_str'].lower() == original_word.lower() and
                                                     x['token_str'] != original_word), mask_candidates))

        # filter out subword tokens
        mask_candidates = list(filter(lambda x: not x['token_str'].startswith("##"), mask_candidates))

        # filter out morphological derivations
        porter_stemmer = PorterStemmer()
        text_lm = porter_stemmer.stem(original_word)
        mask_candidates = \
            list(filter(lambda x: porter_stemmer.stem(x['token_str']) != text_lm
                                  or x['token_str'] == original_word, mask_candidates))

        lemmatizer = WordNetLemmatizer()
        for pos in ["v", "n", "a", "r", "s"]:
            text_lm = lemmatizer.lemmatize(original_word, pos)
            mask_candidates = \
                list(filter(lambda x: lemmatizer.lemmatize(x['token_str'], pos) != text_lm
                                      or x['token_str'] == original_word, mask_candidates))
        # filter out riskset
        mask_candidates = list(filter(lambda x: x['token_str'] not in riskset, mask_candidates))
        # filter out words with any punctuations
        mask_candidates = list(filter(lambda x: not any(s for s in x['token_str'] if s in string.punctuation),
                                      mask_candidates))

        return torch.tensor([x['token'] for x in mask_candidates])



model = InfillModel()
# model.lm_head.from_pretrained("models/ckpt/1101_17:52:48")

model.evaluate()
bf_nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
bf_num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
logger.info(f"Before training: {bf_nli_score, bf_num_subs}")
model.init_metric()

model.train()
model.evaluate()
logger.info(f"Before training: {bf_nli_score, bf_num_subs}")
nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
logger.info(f"After training: {nli_score, num_subs}")

ckpt_dir = f"./models/ckpt/{dt_string}"
model.lm_head.save_pretrained(ckpt_dir)
