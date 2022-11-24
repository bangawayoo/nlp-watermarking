import logging
import os
import random
import sys
sys.path.append(os.path.join(os.getcwd()))

from textattack.transformations import WordInsertionMaskedLM
from augmenter import Augmenter
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
import transformers
from tqdm import tqdm

from utils.misc import get_result_txt




class Attacker:
    def __init__(self, attack_type, attack_percentage, path2txt, path2result, constraint_kwargs, augment_data_flag=False):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        logFormatter = logging.Formatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S")
        log_dir = os.path.dirname(path2txt)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(f"./{log_dir}/corruption.log", mode="w")
        file_handler.setFormatter(logFormatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f"Initializing attacker with [{attack_type}] attack with rate=[{attack_percentage}]")
        self.logger.info(f"Parameters are {constraint_kwargs}")
        self.attack_type = attack_type
        self.num_sentence = constraint_kwargs.get('num_sentence', None)
        use_thres = constraint_kwargs.get("use", 0)
        constraints = [
            UniversalSentenceEncoder(
                threshold=use_thres,
                metric="cosine",
                compare_against_original=True,
                window_size=15,
                skip_text_shorter_than_window=True),
        ]
        self.augmenter_choices = []
        if attack_type == "insertion" or augment_data_flag:
            shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(
                "distilroberta-base" if not augment_data_flag else "bert-large"
            )
            shared_tokenizer = transformers.AutoTokenizer.from_pretrained(
                "distilroberta-base" if not augment_data_flag else "bert-large"
            )
            transformation = WordInsertionMaskedLM(
                masked_language_model=shared_masked_lm,
                tokenizer=shared_tokenizer,
                max_candidates=50,
                min_confidence=0.0,
            )
            self.augmenter_choices.append(Augmenter(transformation=transformation, transformations_per_example=1,
                                                    constraints=constraints,
                                                    pct_words_to_swap=attack_percentage, fast_augment=True))

        elif attack_type == "substitution" or augment_data_flag:
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
            self.augmenter_choices.append(Augmenter(transformation=transformation, transformations_per_example=1,
                                                    constraints=constraints,
                                                    pct_words_to_swap=attack_percentage, fast_augment=True))
        elif attack_type == "char-based":
            pass



        self.result_dir = path2result
        self.wr = open(path2result, "w")
        self.texts = get_result_txt(path2txt)


    def attack_sentence(self, attack_all_samples=True):
        skipped_cnt = 0
        attacked_cnt = 0
        for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_sen, key, msg) in enumerate(tqdm(self.texts)):
            self.logger.info(f"{c_idx} {sen_idx}")
            if attack_all_samples or len(msg) > 0:
                corrupted_sentence = self.augmenter_choices[0].augment(wm_sen)[0]
                self.logger.info(corrupted_sentence)
                self.logger.info(wm_sen)
                if self.attack_type == "insertion" and  len(corrupted_sentence) == len(wm_sen):
                    # corrupted_sentence = "skipped"
                    skipped_cnt += 1
                attacked_cnt += 1

            else:
                corrupted_sentence = wm_sen
                skipped_cnt += 1

            self.wr.write(corrupted_sentence + "\n")
            if self.num_sentence and self.num_sentence == attacked_cnt:
                break

        self.wr.close()
        self.logger.info(f"Skipped {skipped_cnt} and attacked {attacked_cnt} sentences")

    def attack_sample(self):
        sentence_agg = []
        augmented_data = []
        prev_idx = 0

        for idx, (c_idx, sen_idx, sub_idset, sub_idx, wm_sen, key, msg) in enumerate(tqdm(self.texts)):
            if prev_idx == c_idx:
                sentence_agg.append(wm_sen)
                if idx != len(self.texts)-1: # for the last sample, do not continue to the next loop
                    continue

            while True:
                random_sen_idx = random.choice(range(len(sentence_agg)))
                clean_sentence = sentence_agg[random_sen_idx]
                corrupted_sentence = self.augmenter_choices[0].augment(clean_sentence)[0]
                if len(clean_sentence) != len(corrupted_sentence):
                    break

            sentence_agg[random_sen_idx] = corrupted_sentence
            for sen in sentence_agg:
                self.wr.write(sen + "\n")

            self.wr.close()
            augmented_data.extend(sentence_agg)
            sentence_agg = [wm_sen]
            prev_idx = c_idx

    def augment_data(self):
        pass