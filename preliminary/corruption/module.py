from textattack.transformations import WordInsertionMaskedLM
from textattack.augmentation import Augmenter
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
import transformers
import sys


use_thres = 0.95
pct=0.0001 # minimum=1


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

constraints = [
    UniversalSentenceEncoder(
            threshold=use_thres,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True),
]
augmenter = Augmenter(transformation=transformation, transformations_per_example=1, constraints=constraints,
                      pct_words_to_swap=pct)

# load data
data = []
txt_path = "./results/context-ls-result.txt"
with open(txt_path, "r") as f:
    for line in f:
        if line:
            c_idx, w_idx, text = line.split("\t")
            if int(w_idx) == 1:
                data.append(text)

augmented_data = []
for d in data:
    aug_d = augmenter.augment(d)[0]
    if len(aug_d) == len(d):
        aug_d = "Skipped"
    augmented_data.append(aug_d)

with open(f"./results/corrupted-{pct}-watermarked.txt", "w") as f:
    for d in augmented_data:
        f.write(d + "\n")


