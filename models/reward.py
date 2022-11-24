from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM


class NLIReward:
    def __init__(self, device):
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
        self.nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.nli_threshold = 0.95
        self.device = device

    def compute_reward(self, candidate_texts, original_text, probs):
        # run NLI with the original input
        if probs is not None:
            probs = torch.stack(probs, dim=0)
        nli_input = self._concatenate_for_nli(candidate_texts, original_text)
        nli_input = {k: v.to(self.device) for k, v in nli_input.items()}
        with torch.no_grad():
            nli_output = self.nli_model(**nli_input)

        entail_score = nli_output.logits.softmax(dim=-1)[:, 2]
        if probs is not None:
            normalized_prob = probs / probs.sum()
            reward = ((entail_score - self.nli_threshold) * normalized_prob).sum()
            return reward, entail_score
        else:
            return entail_score

    def _concatenate_for_nli(self, candidate_texts, original_text):
        num_samples = len(candidate_texts)
        original_text = [original_text] * num_samples
        nli_input_encoded = self.nli_tokenizer(original_text, candidate_texts, add_special_tokens=True, return_tensors='pt',
                                               padding='longest')

        return nli_input_encoded



class KeywordMatchReward:
    def __init__(self):
        raise NotImplementedError

    # use Jaccard Index
    def compute_reward(self):
        pass
