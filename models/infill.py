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

from config import InfillArgs, riskset
from utils.dataset_utils import preprocess2sentence, preprocess_imdb
from utils.logging import getLogger
from models.reward import NLIReward
from models.kwd import KeywordExtractor
from models.mask import MaskSelector


class InfillModel:
    def __init__(self, args, dirname=None):
        if dirname:
            self.logger = getLogger("INFILL-WATERMARK",
                                    dir_=dirname)
        else:
            self.logger = getLogger("INFILL-WATERMARK")
        self.device = torch.device('cuda')
        self.args = args
        self.train_d, self.test_d = self._init_dataset(args.data_type)
        if args.train_infill:
            self.grad_cnt = 0
            params = [p for p in self.lm_head.parameters()]
            self.optimizer = torch.optim.Adam(params, lr=5e-5)
            self.train_kwargs = {'epoch': 10,
                                 'eval_ep_interval': 1,
                                 'log_iter_interval': 1000
                                 }

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.lm_head = AutoModelForMaskedLM.from_pretrained("bert-base-cased").to(self.device)
        if args.model_ckpt:
            state_dict = torch.load(args.model_ckpt, map_location=self.device)["model"]
            self.lm_head.load_state_dict(state_dict)
        self.nli_reward = None if args.do_watermark else NLIReward(self.device)
        self.keyword_module = KeywordExtractor(ratio=0.06)
        self.mask_selector = MaskSelector()
        self.nlp = spacy.load(args.spacy_model)

        self.metric = {'entail_score': [], 'num_subs': [], 'train_entail_score':[]}
        self.best_metric = {'entail_score': 0}


    def fill_mask(self, text, mask_idx_token, train_flag=True, topk=4, embed_flag=False):
        """
        text: Spacy.Span object (sentence of Spacy.Doc)
        mask_idx_token: List[index of masks in Spacy tokens] (c.f. mask_idx_pt: mask index in Pytorch Tensors)
        """
        tokenized_text = [token.text_with_ws for token in text]
        tokenized_text_masked = tokenized_text.copy()

        # mask all selected tokens
        for m_idx in mask_idx_token:
            tokenized_text_masked[m_idx] = re.sub(r"\S+", self.tokenizer.special_tokens_map['mask_token'],
                                             tokenized_text_masked[m_idx])

        inputs = self.tokenizer(["".join(tokenized_text_masked).strip()], return_tensors="pt",
                                add_special_tokens=True, padding="longest")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        self.logger.debug("Masked Sentence:")
        self.logger.debug("".join(tokenized_text_masked).strip())

        if train_flag:
            logits = self.lm_head(**inputs).logits
        else:
            with torch.no_grad():
                logits = self.lm_head(**inputs).logits

        mask_idx_pt = torch.nonzero(inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=True)

        if mask_idx_pt[0].numel() == 0:
            return [], None, [],[]

        mask_logits = logits[mask_idx_pt]
        probs = mask_logits.softmax(dim=-1)
        sorted_probs, prob_indices = torch.sort(probs, dim=-1, descending=True)

        agg_cwi = []
        agg_probs = []
        valid_input = [True for _ in range(len(mask_idx_token))]
        avg_num_cand = 0

        # filter the chosen tokens
        for idx, m_idx_token in enumerate(mask_idx_token):
            candidate_word_ids = self._filter_words(prob_indices[idx,:32].squeeze(),
                                                        tokenized_text[m_idx_token], embed_flag=embed_flag).to(self.device)

            avg_num_cand += len(candidate_word_ids)
            if len(candidate_word_ids) == 0: #skip this example, if no candidate word survives
                valid_input[idx] = False
                continue
            else:
                candidate_word_ids = candidate_word_ids[:topk]
                # candidate_word_ids = candidate_word_ids

            agg_cwi.append(candidate_word_ids)
            agg_probs.append(sorted_probs[idx, :len(candidate_word_ids)])
            self.logger.debug(self.tokenizer.decode(candidate_word_ids) + "\n")
        avg_num_cand /= len(mask_idx_token)

        return agg_cwi, agg_probs, mask_idx_pt, inputs
        
        
    def generate_candidate_sentence(self, agg_cwi, agg_probs, mask_idx_pt, tokenized_pt):
        candidate_texts = None
        candidate_text_jp = []

        if len(agg_cwi) > 0 :
            candidate_text_ids = []
            # permute through candidate words and probabilities
            for cwi, joint_prob in zip(product(*agg_cwi), product(*agg_probs)):
                compared_inputs = tokenized_pt['input_ids'].clone()
                for m_idx, c_id in zip(mask_idx_pt[1], cwi): #mask_idx_pt is a tuple of pt index; take the second axis's index
                    compared_inputs[:, m_idx] = c_id
                joint_prob = reduce((lambda x,y: x*y), joint_prob)
                candidate_text_jp.append(joint_prob)
                candidate_text_ids.append(compared_inputs)

            candidate_text_ids = torch.cat(candidate_text_ids, dim=0)
            candidate_texts = self.tokenizer.batch_decode(candidate_text_ids, skip_special_tokens=True)

        return candidate_texts, candidate_text_jp


    def compute_nli(self, candidate_texts, candidate_text_jp, text, train_flag=False):
        # run NLI with the original input
        reward, entail_score = self.nli_reward.compute_reward(candidate_texts, text, candidate_text_jp)
        regret = -1 * reward
        if train_flag:
            regret.backward()
            self.grad_cnt += len(entail_score)
        self.logger.debug(candidate_texts)
        self.logger.debug(f"Entailment score {entail_score.mean().item():.3f}")
        # self.logger.info(f"Substituted texts: {self.tokenizer.decode(chosen_word_idx)}")
        # self.logger.info(sum([p.grad.norm() for p in self.lm_head.parameters()]))
        # self.logger.info(entail_score.mean().item())
        if train_flag:
            self.metric['train_entail_score'].extend(entail_score.tolist())
        else:
            self.metric['entail_score'].extend(entail_score.tolist())
            self.metric['num_subs'].append(len(candidate_texts))

        if train_flag and self.grad_cnt >= 128:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.grad_cnt = 0

        return entail_score, candidate_texts


    def run_iter(self, sen, keyword, ent_keyword, train_flag=True, embed_flag=True):
        # mask is selected dependent on the keywords
        mask_idx, mask_word = self.mask_selector.return_mask(sen, keyword, ent_keyword)
        agg_cwi, agg_probs, tokenized_pt, mask_idx_pt = [], [], {}, []

        if mask_idx:
            agg_cwi, agg_probs, mask_idx_pt, tokenized_pt  = self.fill_mask(sen, mask_idx, train_flag=train_flag, embed_flag=embed_flag)

        return agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word)

    def train(self, nli_loss=False, robustness_loss=False):
        iteration = 1
        total_iteration = self.train_kwargs['epoch'] * sum([len(sentences) for sentences in self.train_d])
        for ep in range(self.train_kwargs['epoch']):
            self.logger.info(f"Epoch {ep}")
            for c_idx, sentences in enumerate(self.train_d):
                # extract keyword here
                all_keywords, entity_keywords = self.keyword_module.extract_keyword(sentences)
                for s_idx, sen in enumerate(sentences):
                    iteration += 1
                    self.logger.debug(f"{c_idx} {s_idx}")
                    self.logger.debug(sen)
                    keyword = all_keywords[s_idx]
                    ent_keyword = entity_keywords[s_idx]
                    agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = self.run_iter(sen, keyword, ent_keyword, train_flag=True)
                    candidate_texts, candidate_text_jp = self.generate_candidate_sentence(agg_cwi, agg_probs, mask_idx_pt, tokenized_pt)
                    if nli_loss:
                        entail_score, candidate_texts = self.compute_nli(candidate_texts, candidate_text_jp, sen.text, train_flag=True)
                        self.metric['train_entail_score'].extend(entail_score.tolist())

                    if (iteration+1) % self.train_kwargs['log_iter_interval'] == 0:
                        nli_score = torch.mean(torch.tensor(model.metric['train_entail_score'])).item()
                        self.logger.info(f"Training {iteration}/{total_iteration}: \n NLI: {nli_score:.3f}")
                        self.init_metric()

            if (ep+1) % self.train_kwargs['eval_ep_interval'] == 0:
                self.evaluate(eval_ep=f"{ep}")

        self.logger.info("Saving the final model")
        ckpt_dir = f"./models/ckpt/{args.exp_name}-{dt_string}/final" if args.exp_name else \
            f"./models/ckpt/{dt_string}/final"
        self.lm_head.save_pretrained(ckpt_dir)


    def evaluate(self, eval_ep=""):
        for c_idx, sentences in enumerate(self.test_d):
            # extract keyword here
            all_keywords, entity_keywords = self.keyword_module.extract_keyword(sentences)
            for s_idx, sen in enumerate(sentences):
                # sen_text = sen.text
                self.logger.debug(f"{c_idx} {s_idx}")
                self.logger.debug(sen)
                keyword = all_keywords[s_idx]
                ent_keyword = entity_keywords[s_idx]
                agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = self.run_iter(sen, keyword,
                                                                                                     ent_keyword,
                                                                                                     train_flag=False)
                candidate_texts, candidate_text_jp = self.generate_candidate_sentence(agg_cwi, agg_probs, mask_idx_pt,
                                                                                      tokenized_pt)
                entail_score, candidate_texts = self.compute_nli(candidate_texts, candidate_text_jp, sen.text,
                                                                 train_flag=False)

                self.metric['entail_score'].extend(entail_score.tolist())


        if eval_ep:
            nli_score = torch.mean(torch.tensor(model.metric['entail_score'])).item()
            # num_subs = torch.mean(torch.tensor(model.metric['num_subs'], dtype=float)).item()
            self.logger.info(f"Eval at {eval_ep}: \n NLI: {nli_score:.3f}")
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

        cover_texts = None
        if not self.args.do_watermark:
            cover_texts = preprocess_imdb(corpus)
            cover_texts = preprocess2sentence(cover_texts, dtype+"-train", start_sample_idx, train_num_sample,
                                              spacy_model=self.args.spacy_model)

        test_cover_texts = preprocess_imdb(test_corpus)
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

        # only include those whose first letter is the same case as that of the original word
        def return_case(word):
            word = word.strip()
            if len(word) == 0:
                return False
            return "U" if word[0].isupper() else "l"

        original_case = return_case(original_word)
        if original_case is False:
            return torch.tensor()
        else:
            mask_candidates = list(filter(lambda x: return_case(x['token_str']) == original_case, mask_candidates))

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
    parser = InfillArgs()
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




