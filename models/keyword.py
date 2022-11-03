# Compute keyword score on corpus level.
# Sort them on sentenc level.
# Identify any proper noun or entity per sentence
# If exists, then pick them as keywords first
# if space remains, put the other keywords in descending order

class KeywordExtractor:

    def __init__(self):
        pass

    def extract_keyword(self, sentences):
        """
        Input
         - sentences: (list[Spacy.Span])
         - tokenizer: tokenizer used in NLI to account for truncation
        Output: List[keywords per sentences]
        """

        pass