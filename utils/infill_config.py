from transformers import AutoTokenizer, AutoModelForMaskedLM

INFILL_TOKENIZER = AutoTokenizer.from_pretrained('bert-base-cased')
INFILL_MODEL = AutoModelForMaskedLM.from_pretrained("bert-base-cased")

