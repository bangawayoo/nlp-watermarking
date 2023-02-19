import os
import sys
sys.path.append(os.getcwd())

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import yake

from config import stop


# Find Proper Noun on sentence level.
# Compute Yake on sentence level.
# Compute Tf-idf on sample level

# Sort them on sentence level.
# Identify any proper noun or entity per sentence
# If exists, then pick them as keywords first
# if space remains, put the other keywords in descending order



class KeywordExtractor:

    def __init__(self, **args):
        self.yake_kwargs = {'lan': "en",
                           "n": 1,
                           "dedupLim": 0.9,
                           "dedupFunc": 'seqm',
                           "windowsSize": 1}

        self.args = {}
        for k, v in args.items():
            self.args[k] = v

    def extract_keyword(self, sentences):
        """
        Input
         - sentences: (list[Spacy.Span])
         - tokenizer: tokenizer used in NLI to account for truncation
        Output: List[keywords(Spacy.Token) per sentences]
        """
        all_keywords = []
        entity_keywords = []
        for sen_idx, sen in enumerate(sentences):
            kwd_per_sentence = []
            punct_removed = [token.text for token in sen if not token.is_punct]
            num_kwd = max(1, int(self.args['ratio'] * len(punct_removed)))

            # List[(Token, IOB to determine redundant or not)]
            entity_kwd = self._extract_entity(sen)
            entity_keywords.append([ent[0] for ent in entity_kwd])
            idx = 0
            while len(entity_kwd) > idx and len(kwd_per_sentence) < num_kwd:
                kwd_per_sentence.append(entity_kwd[idx][0])
                if entity_kwd[idx][1] == "I":
                    num_kwd += 1
                idx += 1

            # both has the format List[token, score]
            yake_kwd = self._extract_yake_kwd(sen)
            yake_kwd = [s[0] for s in yake_kwd]
            # tfidf_kwd = tfidf_keywords[sen_idx]
            # tfidf_kwd = []
            # kwd = self._normalize_scores(yake_kwd)

            idx = 0
            while len(yake_kwd) > idx and len(kwd_per_sentence) < num_kwd:
                kwd_per_sentence.append(yake_kwd[idx])
                idx += 1

            all_keywords.append(kwd_per_sentence)

        return all_keywords, entity_keywords

    def _normalize_scores(self, *scores):
        normalized_kwd = []
        # for score in scores:
        #     kwd = [i[0] for i in score]
        #     score = [i[1] for i in score]
        #     score = [s / sum(score) for s in score]
        pass

    def _extract_yake_kwd(self, sentence):
        output = []
        sen_text = sentence.text
        kw_extractor = yake.KeywordExtractor(**self.yake_kwargs, top=20)
        keyword = kw_extractor.extract_keywords(sen_text)  # list  of (phrase: score)
        keyword_list = [k[0] for k in keyword]

        for token in sentence:
            if token.text in keyword_list and token.pos_ in ["NOUN", "PROPN"]:
                idx = keyword_list.index(token.text)
                output.append([token, keyword[idx][1]])
                keyword_list.remove(token.text)
        return output

    def _extract_entity(self, sentence):
        extracted = []
        for token in sentence:
            if token.ent_type_ in ["GPE", "ORG", "PERSON", "WORK_OF_ART", "EVENT"]:
                extracted.append((token, token.ent_iob_))

        return extracted

    def _extract_tfidf(self, sentences):
        from datasets import load_dataset

        output = []
        sample = load_dataset("imdb")['train']['text'][0]
        nlp = spacy.load("en_core_web_sm")
        sentences = [s for s in nlp(sample).sents]
        input2vectorizer = [[token.text for token in sent] for sent in sentences]

        tfidf = TfidfVectorizer(analyzer=lambda x: [w for w in x if w not in stop])
        tfidf_vectors = tfidf.fit_transform(input2vectorizer)
        # indptr = tfidf_vectors.indptr
        # indices = tfidf_vectors.indices
        # data = tfidf_vectors.data

        tfidf_vectors = tfidf_vectors.todense()
        features = tfidf.get_feature_names_out()
        # for row in tfidf_vectors:

        return



if __name__ == "__main__":
    from datasets import load_dataset
    from utils.misc import preprocess2sentence, preprocess_imdb

    dtype = 'imdb'
    start_sample_idx = 0
    corpus = load_dataset(dtype)['train']['text']
    spacy_model = "en_core_web_sm"
    train_num_sample = 100

    cover_texts = [t.replace("\n", " ") for t in corpus]
    cover_texts = preprocess_imdb(cover_texts)
    cover_texts = preprocess2sentence(cover_texts, dtype + "-train", start_sample_idx, train_num_sample,
                                      spacy_model=spacy_model)

    kwd_module = KeywordExtractor()
    keyword = kwd_module.extract_keyword(cover_texts[0])
    # breakpoint()