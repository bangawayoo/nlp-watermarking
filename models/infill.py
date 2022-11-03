from datetime import datetime
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import sys
sys.path.append(os.getcwd())

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM

from config import InfillArgs
from dataset_utils import preprocess2sentence, preprocess_imdb
from utils import *
from reward import NLIReward

parser = InfillArgs()
args = parser.parse_args()

now = datetime.now()
dt_string = now.strftime("%m%d_%H%M%S")
stop = set(stopwords.words('english'))
punctuation = set(string.punctuation)
riskset = stop.union(punctuation)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")

log_path = f"./logs/infill/{args.exp_name}-{dt_string}.log" if args.exp_name \
    else f"./logs/infill/{dt_string}.log"

log_dir = os.path.dirname(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not args.debug_mode:
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logFormatter)
    logger.addHandler(file_handler)

streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG if args.debug_mode else logging.INFO)
logger.addHandler(streamHandler)

class InfillModel:
    def __init__(self, args):
        self.device = torch.device('cuda')
        self.args = args
        self.train_d, self.test_d = self._init_dataset(args.data_type)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        if args.model_ckpt is None:
            ckpt = "bert-base-cased"
        self.lm_head = AutoModelForMaskedLM.from_pretrained(ckpt).to(self.device)
        self.nli_reward = NLIReward(self.device)
        self.grad_cnt = 0
        params = [p for p in self.lm_head.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=5e-5)
        self.metric = {'entail_score':[], 'num_subs':[]}
        self.best_metric = {'entail_score': 0}
        self.kwd_kwargs = {'lan': "en",
                           "n": 1,
                           "dedupLim": 0.9,
                           "dedupFunc": 'seqm',
                           "windowsSize": 1}
        self.train_kwargs = {'epoch':10,
                             'eval_ep_interval':1,
                             'log_iter_interval':100
                             }

    def fill_mask(self, text, mask_indices, train_flag=True, topk=2):
        """
        text: Spacy.Span object (sentence of Spacy.Doc)
        """
        offset = text[0].i
        mask_indices = [m-offset for m in mask_indices]
        tokenized_text = [token.text for token in text]
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

        candidate_text_ids = []
        valid_input = [True for _ in range(len(mask_indices))]
        avg_num_cand = 0
        # filter the chosen tokens
        for idx, m_idx in enumerate(mask_indices):
            candidate_word_ids = self._filter_words(indices[idx,:32].squeeze(), tokenized_text[m_idx]).to(self.device)
            avg_num_cand += len(candidate_word_ids)
            if len(candidate_word_ids) == 0:
                # candidate_word_ids = inputs['input_ids'][idx, m_idx].clone().repeat(topk)
                valid_input[idx] = False
                continue
            elif len(indices) < topk:
                candidate_word_ids = candidate_word_ids.repeat(topk)[:topk]
            else:
                candidate_word_ids = candidate_word_ids[:topk]

            compared_inputs = inputs['input_ids'][idx].clone().repeat(topk, 1) # (batch, seq)
            compared_inputs[:, m_idx] = candidate_word_ids
            candidate_text_ids.append(compared_inputs)
        avg_num_cand /= len(mask_indices)

        if len(candidate_text_ids) > 0 :
            candidate_text_ids = torch.cat(candidate_text_ids, dim=0)
            candidate_texts = self.tokenizer.batch_decode(candidate_text_ids, skip_special_tokens=True)
            # run NLI with the original input
            reward, entail_score = self.nli_reward.compute_reward(candidate_texts, text.text, sorted_probs[valid_input, :topk].flatten())
            regret = -1 * reward
            if train_flag:
                regret.backward()
            self.grad_cnt += len(entail_score)

            logger.debug(f"Masked text: {[tokenized_text[m_idx]  for m_idx in mask_indices]}")
            # logger.info(f"Substituted texts: {self.tokenizer.decode(chosen_word_idx)}")
            # logger.info(sum([p.grad.norm() for p in self.lm_head.parameters()]))
            # logger.info(entail_score.mean().item())
            self.metric['entail_score'].extend(entail_score.tolist())
            self.metric['num_subs'].append(avg_num_cand)

        if self.grad_cnt >= 128 and train_flag:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.grad_cnt = 0

    def run_iter(self, mode, **input):
        sen = input['sen']
        sen_text = sen.text
        logger.debug(sen_text)
        # truncate text to the max length limit (minus 10 to be safe)
        sen_text = tokenizer.decode(tokenizer(sen_text, add_special_tokens=False, truncation=True,
                                              max_length=tokenizer.model_max_length // 2 - 10)['input_ids'])

        punct_removed = [token.text for token in sen if not token.is_punct]

        word_count = len(punct_removed)
        num_kwd = max(1, int(0.15 * word_count))
        # pick keywords
        kw_extractor = yake.KeywordExtractor(**self.kwd_kwargs, top=20)
        # truncate text to the max length limit (minus 10 to be safe)
        keywords = kw_extractor.extract_keywords(sen_text)  # dict of {phrase: score}
        selected_keywords = [k[0] for k in keywords[:num_kwd]]
        kw_hash = set(selected_keywords)

        masked = []

        for token in sen:
            if token.text in selected_keywords:
                masked.append((token.head, token.head.i))
                selected_keywords.remove(token.text)

        if masked:
            mask_indices = [i[1] for i in masked]
            model.fill_mask(sen, mask_indices, train_flag=True if mode=="train" else False)

    def train(self):
        iteration = 1
        for ep in range(self.train_kwargs['epoch']):
            logger.info(f"Epoch {ep}")
            for c_idx, sentences in enumerate(self.train_d):
                # extract keyword here
                for s_idx, sen in enumerate(sentences):
                    iteration += 1
                    sen_text = sen.text
                    logger.debug(f"{c_idx} {s_idx}")
                    logger.debug(sen)
                    # truncate text to the max length limit (minus 10 to be safe)
                    sen_text = tokenizer.decode(tokenizer(sen_text, add_special_tokens=False, truncation=True,
                                                     max_length=tokenizer.model_max_length // 2 - 10)['input_ids'])

                    punct_removed = [token.text for token in sen if not token.is_punct]

                    word_count = len(punct_removed)
                    num_kwd = max(1, int(0.15 * word_count))
                    # pick keywords
                    kw_extractor = yake.KeywordExtractor(**self.kwd_kwargs, top=20)
                    # truncate text to the max length limit (minus 10 to be safe)
                    keywords = kw_extractor.extract_keywords(sen_text)  # dict of {phrase: score}
                    selected_keywords = [k[0] for k in keywords[:num_kwd]]
                    kw_hash = set(selected_keywords)

                    masked = []

                    for token in sen:
                        if token.text in selected_keywords:
                            masked.append((token.head, token.head.i))
                            selected_keywords.remove(token.text)

                    if masked:
                        mask_indices = [i[1] for i in masked]
                        model.fill_mask(sen, mask_indices)

                    # if self.train_kwargs['log_iter_interval'] % iteration == 0:
                    #     logger.info(f"entail score: {entail_score.mean().item()}")

            if self.train_kwargs['eval_ep_interval'] % (ep+1) == 0:
                self.evaluate(eval_ep=f"{ep}")


    def evaluate(self, eval_ep=""):
        for c_idx, sentences in enumerate(self.test_d):
            for s_idx, sen in enumerate(sentences):
                sen_text = sen.text
                logger.debug(f"{c_idx} {s_idx}")
                logger.debug(sen)
                # truncate text to the max length limit (minus 10 to be safe)
                sen_text = tokenizer.decode(tokenizer(sen_text, add_special_tokens=False, truncation=True,
                                                 max_length=tokenizer.model_max_length // 2 - 10)['input_ids'])

                punct_removed = [token.text for token in sen if not token.is_punct]
                word_count = len(punct_removed)
                num_kwd = max(1, int(0.15 * word_count))

                # pick keywords
                kw_extractor = yake.KeywordExtractor(**self.kwd_kwargs, top=20)
                keywords = kw_extractor.extract_keywords(sen_text)  # dict of {phrase: score}
                selected_keywords = [k[0] for k in keywords[:num_kwd]]
                kw_hash = set(selected_keywords)
                masked = []

                for token in sen:
                    if token.text in selected_keywords:
                        masked.append((token.head, token.head.i))
                        selected_keywords.remove(token.text)

                if masked:
                    mask_indices = [i[1] for i in masked]
                    model.fill_mask(sen, mask_indices, train_flag=False)
        if eval_ep:
            nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
            num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
            logger.info(f"Eval at {eval_ep}: \n NLI: {nli_score:.3f}, # of candidates:{num_subs:1f}")
            if self.best_metric['entail_score'] < nli_score:
                self.best_metric['entail_score'] = nli_score
                ckpt_dir = f"./models/ckpt/{dt_string}/best"
                self.lm_head.save_pretrained(ckpt_dir)
            model.init_metric()

    def init_metric(self):
        self.metric = {'entail_score': [], 'num_subs': []}

    def _init_dataset(self, dtype="imdb"):
        start_sample_idx = 0
        if dtype == "imdb":
            corpus = load_dataset(dtype)['train']['text']
            test_corpus = load_dataset(dtype)['test']['text']
        else:
            raise NotImplementedError

        train_num_sample = len(corpus)
        test_num_sample = 1000

        cover_texts = [t.replace("\n", " ") for t in corpus]
        cover_texts = preprocess_imdb(cover_texts)
        cover_texts = preprocess2sentence(cover_texts, dtype+"-train", start_sample_idx, train_num_sample,
                                          spacy_model=self.args.spacy_model)

        test_cover_texts = [t.replace("\n", " ") for t in test_corpus]
        test_cover_texts = preprocess_imdb(test_cover_texts)
        test_cover_texts = preprocess2sentence(test_cover_texts, dtype+"-test", start_sample_idx, test_num_sample,
                                               spacy_model=self.args.spacy_model)
        return cover_texts, test_cover_texts


    def _filter_words(self, candidate_word_ids, original_word):
        mask_candidates = [{'token_str': self.tokenizer.decode(x),
                            'token': x}
                           for x in candidate_word_ids]
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


if __name__ == "__main__":
    model = InfillModel(args)

    # model.evaluate()
    # bf_nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
    # bf_num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
    # logger.info(f"Before training: {bf_nli_score, bf_num_subs}")
    # model.init_metric()

    if not args.eval_only:
        model.train()
        ckpt_dir = f"./models/ckpt/{args.exp_name}-{dt_string}" if args.exp_name else \
            f"./models/ckpt/{dt_string}"
        model.lm_head.save_pretrained(ckpt_dir)

    model.evaluate()
    nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
    num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
    logger.info(f"After training: {nli_score, num_subs}")




