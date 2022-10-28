import os
import sys
sys.path.append(os.path.join(os.getcwd()))
from tqdm import tqdm
import logging

from utils import get_result_txt

from textattack.transformations import WordInsertionMaskedLM
from textattack.augmentation import Augmenter
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
import transformers

import random
# os.chdir("./Research/NLP/nlp-watermarking/preliminary")


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(f"./logs/corruption.log", mode="a")
file_handler.setFormatter(logFormatter)
logger.addHandler(file_handler)


# load data
dtype = "imdb"
num_sample = 50

# corpus = arxiv_cs_abstracts("train",  attrs=['abstract'])
result_dir = f"results/context-ls-{dtype}-corrupted={pct}.txt"
wr = open(result_dir, "a")
watermarked_samples = get_result_txt(f"./results/context-ls-{dtype}-0-100.txt")



class Attacker:
    def __init__(self, attack_type, attack_percentage, path2txt, path2result, constraint_kwargs):
        if attack_type == "insertion":
            shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(
                "distilroberta-base"
            )
            shared_tokenizer = transformers.AutoTokenizer.from_pretrained(
                "distilroberta-base"
            )
            transformation = WordInsertionMaskedLM(
                masked_language_model=shared_masked_lm,
                tokenizer=shared_tokenizer,
                max_candidates=50,
                min_confidence=0.0,
            )
        elif attack_type == "substitution":
            pass
        elif attack_type == "char-based":
            pass

        use_thres = constraint_kwargs.get("use", 0)
        # use_thres = 0.95
        # attack_percentage = 0.0001
        constraints = [
            UniversalSentenceEncoder(
                threshold=use_thres,
                metric="cosine",
                compare_against_original=True,
                window_size=15,
                skip_text_shorter_than_window=True),
        ]

        self.result_dir = path2result
        self.wr = open(path2result, "w")
        self.texts = get_result_txt(path2txt)
        self.augmenter = Augmenter(transformation=transformation, transformations_per_example=1, constraints=constraints,
                              pct_words_to_swap=attack_percentage, fast_augment=True)

    def attack_sentence(self):
        skipped_cnt = 0
        for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_sen, key, msg) in enumerate(tqdm(watermarked_samples)):
            if msg:
                corrupted_sentence = self.augmenter.augment(wm_sen)[0]
                logger.info(len(corrupted_sentence) == len(wm_sen))
                logger.info(corrupted_sentence)
                logger.info(wm_sen)
                if len(corrupted_sentence) == len(wm_sen):
                    corrupted_sentence = "skipped"
                    skipped_cnt += 1
            else:
                corrupted_sentence = wm_sen

            self.wr.write(corrupted_sentence + "\n")

        logger.info(f"Skipped {skipped_cnt} out of {len(watermarked_samples)} sentences")

    def attack_sample(self):
        sentence_agg = []
        augmented_data = []
        prev_idx = 0

        for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_sen, key, msg) in enumerate(tqdm(watermarked_samples)):
            if prev_idx == c_idx:
                sentence_agg.append(wm_sen)
                if idx != len(watermarked_samples)-1: # for the last sample, do not continue to next loop
                    continue

            while True:
                random_sen_idx = random.choice(range(len(sentence_agg)))
                clean_sentence = sentence_agg[random_sen_idx]
                corrupted_sentence = self.augmenter.augment(clean_sentence)[0]
                if len(clean_sentence) != len(corrupted_sentence):
                    break

            sentence_agg[random_sen_idx] = corrupted_sentence
            for sen in sentence_agg:
                self.wr.write(sen + "\n")


            augmented_data.extend(sentence_agg)
            sentence_agg = [wm_sen]
            prev_idx = c_idx
