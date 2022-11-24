from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch

from utils.dataset_utils import preprocess2sentence, preprocess_imdb
from utils.misc import get_result_txt

class Metric:
    def __init__(self, **kwargs):
        self.sts_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.args = {}
        for k,v in kwargs.items():
            self.args[k] = v


    def compute_ss(self, path2wm, cover_texts=None):
        if cover_texts is None:
            corpus = load_dataset(self.args['dtype'])['test']['text']
            cover_texts = [t.replace("\n", " ") for t in corpus]
            cover_texts = preprocess_imdb(cover_texts)
            cover_texts = preprocess2sentence(cover_texts, self.args['dtype'] + "-test", 0, self.args['num_sample'],
                                              spacy_model=self.args['spacy_model'])

        watermarked = get_result_txt(path2wm)
        batch = []
        wm_batch = []
        sr_score = []

        for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_text, key, msg) in enumerate(watermarked):
            if len(msg) > 0:
                text = cover_texts[int(c_idx)][int(sen_idx)].text
                batch.append(text)
                wm_batch.append(wm_text.strip())

            if len(batch) == 64 or idx == len(watermarked)-1:
                wm_emb = self.sts_model.encode(wm_batch, convert_to_tensor=True)
                emb = self.sts_model.encode(batch, convert_to_tensor=True)
                cosine_scores = util.cos_sim(wm_emb, emb)
                sr_score.extend(torch.diagonal(cosine_scores).tolist())
                batch = []
                wm_batch = []
        return sr_score

