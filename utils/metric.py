import os.path

from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.dataset_utils import preprocess2sentence, preprocess_txt, get_result_txt


class Metric:
    def __init__(self, device, **kwargs):
        self.sts_model = {"all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2'),
                          "roberta": SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')}
        self.awt_model = SentenceTransformer('bert-base-nli-mean-tokens')
        # self.nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-large').to(device)
        # self.nli_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-large')
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
        self.nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large')

        self.device = device

        self.args = {}
        self.test_cv = None
        for k, v in kwargs.items():
            self.args[k] = v

    def load_test_cv(self):
        corpus = load_dataset(self.args['dtype'])['test']['text']
        cover_texts = [t.replace("\n", " ") for t in corpus]
        cover_texts = preprocess_txt(cover_texts)
        cover_texts = preprocess2sentence(cover_texts, self.args['dtype'] + "-test", 0, self.args['num_sample'],
                                          spacy_model=self.args['spacy_model'])
        self.test_cv = cover_texts
        return cover_texts

    def compute_ss(self, path2wm, model_name, cover_texts=None):
        if cover_texts is None:
            cover_texts = self.test_cv if self.test_cv else self.load_test_cv()

        watermarked = get_result_txt(path2wm)
        batch = []
        wm_batch = []
        sr_score = []
        sr_dist = []
        for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_text, key, msg) in enumerate(watermarked):
            # only include watermarks that are altered from the original
            if len(msg) > 0:
                text = cover_texts[int(c_idx)][int(sen_idx)].text
                batch.append(text)
                wm_batch.append(wm_text.strip())
            if (len(batch) == 64 or idx == len(watermarked) - 1) and len(wm_batch) != 0:
                wm_emb = self.sts_model[model_name].encode(wm_batch, convert_to_tensor=True)
                emb = self.sts_model[model_name].encode(batch, convert_to_tensor=True)
                cosine_scores = util.cos_sim(wm_emb, emb)
                sr_score.extend(torch.diagonal(cosine_scores).tolist())
                # wm_emb = self.awt_model.encode(wm_batch)
                # emb = self.awt_model.encode(batch)
                l2dist = np.linalg.norm(wm_emb.cpu() - emb.cpu(), axis=1)
                # cosine_scores = util.cos_sim(wm_emb, emb)
                sr_score.extend(torch.diagonal(cosine_scores).tolist())
                sr_dist.extend(l2dist.tolist())
                batch = []
                wm_batch = []
        return sr_score, sr_dist

    def compute_nli(self, path2wm, cover_texts=None):
        if cover_texts is None:
            cover_texts = self.test_cv if self.test_cv else self.load_test_cv()

        watermarked = get_result_txt(path2wm)
        batch = []
        wm_batch = []
        nli_score = []
        all_texts = []
        all_wm_texts = []

        for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_text, key, msg) in enumerate(watermarked):
            # only include watermarks that are altered from the original
            if len(msg) > 0:
                text = cover_texts[int(c_idx)][int(sen_idx)].text
                batch.append(text)
                wm_batch.append(wm_text.strip())

            if (len(batch) == 64 or idx == len(watermarked) - 1) and len(wm_batch) != 0:
                nli_encodings = self._concatenate_for_nli(batch, wm_batch)
                nli_encodings = {k: v.to(self.device) for k, v in nli_encodings.items()}
                with torch.no_grad():
                    scores = self.nli_model(**nli_encodings).logits
                    entail_score = torch.nn.functional.softmax(scores, dim=-1)[:, 2]

                # scores = self.nli_model.predict(batch)
                # if len(scores.shape) == 1:
                #     scores = scores[np.newaxis, :]
                # e_x = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                # entail_score = (e_x / np.sum(e_x, axis=1, keepdims=True))[:, 1]
                all_texts.extend(batch)
                all_wm_texts.extend(wm_batch)
                nli_score.extend(entail_score.tolist())
                batch = []
                wm_batch = []

        return nli_score, all_texts, all_wm_texts

    def _concatenate_for_nli(self, candidate_texts, original_texts):
        nli_input_encoded = self.nli_tokenizer(original_texts, candidate_texts, add_special_tokens=True,
                                               return_tensors='pt',
                                               padding='longest')

        return nli_input_encoded
