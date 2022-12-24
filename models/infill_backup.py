import logging
from datetime import datetime
import os
import sys
sys.path.append(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import string
from itertools import product
from functools import reduce
import re

from datasets import load_dataset
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import spacy

from config import WatermarkArgs, riskset
from utils.dataset_utils import preprocess2sentence, preprocess_txt
from utils.logging import getLogger
from models.reward import NLIReward
from models.kwd import KeywordExtractor
from models.mask import MaskSelector

logger = getLogger("INFILL-WATERMARK")


class InfillModel:
    def __init__(self, args):
        self.device = torch.device('cuda')
        self.args = args
        self.train_d, self.test_d = self._init_dataset(args.data_type)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        ckpt = "bert-base-cased" if args.model_ckpt is None else args.model_ckpt
        self.lm_head = AutoModelForMaskedLM.from_pretrained(ckpt).to(self.device)
        self.nli_reward = NLIReward(self.device)
        self.keyword_module = KeywordExtractor(ratio=0.05)
        self.mask_selector = MaskSelector()
        self.nlp = spacy.load(args.spacy_model)
        self.grad_cnt = 0
        params = [p for p in self.lm_head.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=5e-5)
        self.metric = {'entail_score':[], 'num_subs':[], 'train_entail_score':[]}
        self.best_metric = {'entail_score': 0}
        self.train_kwargs = {'epoch':10,
                             'eval_ep_interval':1,
                             'log_iter_interval':1000
                             }

    def fill_mask(self, text, mask_idx_str, train_flag=True, topk=4, embed_flag=False):
        """
        text: Spacy.Span object (sentence of Spacy.Doc)
        """
        tokenized_text = [token.text_with_ws for token in text]
        tokenized_text_masked = tokenized_text.copy()

        # mask all selected tokens
        for m_idx in mask_idx_str:
            tokenized_text_masked[m_idx] = re.sub(r"\S+", self.tokenizer.special_tokens_map['mask_token'],
                                             tokenized_text_masked[m_idx])

        inputs = self.tokenizer(["".join(tokenized_text_masked).strip()], return_tensors="pt",
                                add_special_tokens=True, padding="longest")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        logger.debug("".join(tokenized_text_masked).strip())

        if train_flag:
            logits = self.lm_head(**inputs).logits
        else:
            with torch.no_grad():
                logits = self.lm_head(**inputs).logits

        masked_idx_pt = torch.nonzero(inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=True)

        if masked_idx_pt[0].numel()==0 :
            return [], None

        mask_logits = logits[masked_idx_pt]
        probs = mask_logits.softmax(dim=-1)
        sorted_probs, prob_indices = torch.sort(probs, dim=-1, descending=True)

        agg_cwi = []
        agg_probs = []
        valid_input = [True for _ in range(len(mask_idx_str))]
        avg_num_cand = 0

        # filter the chosen tokens
        for idx, m_idx_str in enumerate(mask_idx_str):
            candidate_word_ids = self._filter_words(prob_indices[idx,:32].squeeze(),
                                                        tokenized_text[m_idx_str], embed_flag=embed_flag).to(self.device)

            avg_num_cand += len(candidate_word_ids)
            if len(candidate_word_ids) == 0:
                # candidate_word_ids = inputs['input_ids'][idx, m_idx].clone().repeat(topk)
                valid_input[idx] = False
                continue
            else:
                candidate_word_ids = candidate_word_ids[:topk]
                # candidate_word_ids = candidate_word_ids

            # compared_inputs = inputs['input_ids'].clone().repeat(topk, 1) # (batch, seq)
            agg_cwi.append(candidate_word_ids)
            agg_probs.append(sorted_probs[idx, :len(candidate_word_ids)])
            # compared_inputs[:, masked_idx_pt[1][idx]] = candidate_word_ids
            logger.debug(self.tokenizer.decode(candidate_word_ids))
            # candidate_text_ids.append(compared_inputs)
        avg_num_cand /= len(mask_idx_str)

        if embed_flag:
            return agg_cwi, mask_idx_str

        entail_score = None
        candidate_texts = None
        if len(agg_cwi) > 0 :
            candidate_text_ids = []
            candidate_text_jp = []

            # permute through candidate words and probabilities
            for cwi, joint_prob in zip(product(*agg_cwi), product(*agg_probs)):
                compared_inputs = inputs['input_ids'].clone()
                for m_idx, c_id in zip(masked_idx_pt[1], cwi): #masked_idx_pt is a tuple of pt index; take the second axis's index
                    compared_inputs[:, m_idx] = c_id
                joint_prob = reduce((lambda x,y: x*y), joint_prob)
                candidate_text_jp.append(joint_prob)
                candidate_text_ids.append(compared_inputs)

            candidate_text_ids = torch.cat(candidate_text_ids, dim=0)
            candidate_texts = self.tokenizer.batch_decode(candidate_text_ids, skip_special_tokens=True)
            # run NLI with the original input
            reward, entail_score = self.nli_reward.compute_reward(candidate_texts, text.text, candidate_text_jp)
            regret = -1 * reward
            if train_flag:
                regret.backward()
            self.grad_cnt += len(entail_score)
            logger.debug(candidate_texts)
            logger.debug(f"Entailment score {entail_score.mean().item():.3f}")
            logger.debug(f"Masked text: {[tokenized_text[m_idx]  for m_idx in mask_idx_str]}\n")
            # logger.info(f"Substituted texts: {self.tokenizer.decode(chosen_word_idx)}")
            # logger.info(sum([p.grad.norm() for p in self.lm_head.parameters()]))
            # logger.info(entail_score.mean().item())
            if train_flag:
                self.metric['train_entail_score'].extend(entail_score.tolist())
            else:
                self.metric['entail_score'].extend(entail_score.tolist())
                self.metric['num_subs'].append(avg_num_cand)

        if self.grad_cnt >= 128 and train_flag:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.grad_cnt = 0

        return entail_score, candidate_texts

    def run_iter(self, sen, keyword, ent_keyword, train_flag=True, embed_flag=False):
        # mask is selected dependent on the keywords
        mask_idx, mask_word = self.mask_selector.return_mask(sen, keyword, ent_keyword)
        candidate_word_ids = []
        if mask_idx:
            candidate_word_ids, mask_idx = self.fill_mask(sen, mask_idx, train_flag=train_flag, embed_flag=embed_flag)

        if embed_flag:
            return candidate_word_ids, mask_idx, mask_word

    def train(self):
        iteration = 1
        total_iteration = self.train_kwargs['epoch'] * sum([len(sentences) for sentences in self.train_d])
        for ep in range(self.train_kwargs['epoch']):
            logger.info(f"Epoch {ep}")
            for c_idx, sentences in enumerate(self.train_d):
                # extract keyword here
                all_keywords, entity_keywords = self.keyword_module.extract_keyword(sentences)
                for s_idx, sen in enumerate(sentences):
                    iteration += 1
                    logger.debug(f"{c_idx} {s_idx}")
                    logger.debug(sen)
                    keyword = all_keywords[s_idx]
                    ent_keyword = entity_keywords[s_idx]
                    self.run_iter(sen, keyword, ent_keyword, train_flag=True)

                    if (iteration+1) % self.train_kwargs['log_iter_interval'] == 0:
                        nli_score = torch.mean(torch.tensor(model.metric['train_entail_score'])).item()
                        logger.info(f"Training {iteration}/{total_iteration}: \n NLI: {nli_score:.3f}")
                        self.init_metric()

            if (ep+1) % self.train_kwargs['eval_ep_interval'] == 0:
                self.evaluate(eval_ep=f"{ep}")

        logger.info("Saving the final model")
        ckpt_dir = f"./models/ckpt/{args.exp_name}-{dt_string}/final" if args.exp_name else \
            f"./models/ckpt/{dt_string}/final"
        self.lm_head.save_pretrained(ckpt_dir)


    def evaluate(self, eval_ep=""):
        for c_idx, sentences in enumerate(self.test_d):
            # extract keyword here
            all_keywords, entity_keywords = self.keyword_module.extract_keyword(sentences)
            for s_idx, sen in enumerate(sentences):
                # sen_text = sen.text
                logger.debug(f"{c_idx} {s_idx}")
                logger.debug(sen)
                keyword = all_keywords[s_idx]
                ent_keyword = entity_keywords[s_idx]
                self.run_iter(sen, keyword, ent_keyword, train_flag=False)

        if eval_ep:
            nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
            num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
            logger.info(f"Eval at {eval_ep}: \n NLI: {nli_score:.3f}, # of candidates:{num_subs:1f}")
            if self.best_metric['entail_score'] < nli_score:
                self.best_metric['entail_score'] = nli_score
                ckpt_dir = f"./models/ckpt/{self.args.exp_name}-{dt_string}/best" if self.args.exp_name else \
                    f"./models/ckpt/{dt_string}/best"
                self.lm_head.save_pretrained(ckpt_dir)
            model.init_metric()

    def init_metric(self):
        self.metric = {'entail_score': [], 'num_subs': [], 'train_entail_score': []}

    def _init_dataset(self, dtype="imdb"):
        start_sample_idx = 0
        if dtype == "imdb":
            corpus = load_dataset(dtype)['train']['text']
            test_corpus = load_dataset(dtype)['test']['text']
        else:
            raise NotImplementedError

        if self.args.debug_mode:
            train_num_sample = 10
            test_num_sample = 10
        else:
            train_num_sample = len(corpus)
            test_num_sample = 1000

        cover_texts = preprocess_txt(corpus)
        cover_texts = preprocess2sentence(cover_texts, dtype+"-train", start_sample_idx, train_num_sample,
                                          spacy_model=self.args.spacy_model)

        test_cover_texts = [t.replace("\n", " ") for t in test_corpus]
        test_cover_texts = preprocess_txt(test_cover_texts)
        test_cover_texts = preprocess2sentence(test_cover_texts, dtype+"-test", start_sample_idx, test_num_sample,
                                               spacy_model=self.args.spacy_model)
        return cover_texts, test_cover_texts

    def return_dataset(self):
        return self.train_d, self.test_d

    def _filter_words(self, candidate_word_ids, original_word, embed_flag=False):
        mask_candidates = [{'token_str': self.tokenizer.decode(x),
                            'token': x}
                           for x in candidate_word_ids]

        # filter out subword tokens
        mask_candidates = list(filter(lambda x: not x['token_str'].startswith("##"), mask_candidates))

        # filter out riskset
        mask_candidates = list(filter(lambda x: x['token_str'] not in riskset, mask_candidates))
        # filter out words with any punctuations
        mask_candidates = list(filter(lambda x: not any(s for s in x['token_str'] if s in string.punctuation),
                                      mask_candidates))


        # filtering based on the original word is only possible when training
        if not embed_flag:
            # filter out morphological derivations
            mask_candidates = list(filter(lambda x: not (x['token_str'].lower() == original_word.lower() and
                                                         x['token_str'] != original_word), mask_candidates))
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

        return torch.tensor([x['token'] for x in mask_candidates])


if __name__ == "__main__":
    parser = WatermarkArgs()
    args = parser.parse_args()
    model = InfillModel(args)
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")
    logger = getLogger("INFILL")

    model.evaluate()
    bf_nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
    bf_num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
    logger.info(f"Before training: {bf_nli_score, bf_num_subs}")
    model.init_metric()

    if not args.eval_only:
        model.train()

        ckpt_dir = f"./models/ckpt/{args.exp_name}-{dt_string}/final" if args.exp_name else \
            f"./models/ckpt/{dt_string}/final"
        model.lm_head.save_pretrained(ckpt_dir)

    model.evaluate()
    nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
    num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
    logger.info(f"After training: {nli_score, num_subs}")




