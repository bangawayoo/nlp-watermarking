import random
from string import punctuation

class MaskSelector:
    def __init__(self, **kwargs):
        self.kwargs = {}
        for k, v in kwargs.items():
            self.kwargs[k] = v

        self.num_max_mask = []
        self.custom_keywords = kwargs.get("custom_keywords", [])
        self.dep_ordering = ['expl', 'cc', 'auxpass', 'agent', 'mark', 'aux', 'prep', 'det', 'prt', 'intj', 'parataxis',
                             'predet', 'case', 'csubj', 'acl', 'advcl', 'ROOT', 'preconj', 'ccomp', 'relcl', 'advmod',
                             'dative', 'xcomp', 'pcomp', 'nsubj', 'quantmod', 'conj', 'nsubjpass', 'punct', 'poss',
                             'dobj', 'nmod', 'attr', 'csubjpass', 'meta', 'pobj', 'amod', 'npadvmod', 'appos', 'acomp',
                             'compound', 'oprd', 'nummod']

        if self.kwargs.get('exclude_cc', False):
            self.dep_ordering.remove("cc")
            self.dep_ordering = self.dep_ordering[:15] + ["cc"]
        else:
            self.dep_ordering = self.dep_ordering[:15]


        self.pos_ordering = ['CCONJ', 'AUX', 'ADP', 'SCONJ', 'DET', 'SPACE', 'INTJ', 'PRON', 'SYM', 'VERB', 'ADV',
                             'PUNCT', 'X', 'PART', 'NOUN', 'ADJ', 'PROPN', 'NUM']

    def return_mask(self, sen, keyword, ent_keyword):
        """
        keyword: List[Spacy tokens]
        """
        if self.kwargs['method'] == "keyword_disconnected":
            return self.keyword_disconnected(sen, keyword, ent_keyword)
        elif self.kwargs['method'] == "keyword_connected":
            return self.keyword_connected(sen, keyword, ent_keyword, type=self.kwargs['keyword_mask'])
        elif self.kwargs['method'] == "grammar":
            return self.grammar_component(sen, keyword, ent_keyword, ordering_by=self.kwargs['mask_order_by'])

    def keyword_connected(self, sen, keyword, ent_keyword, type="adjacent"):
        mask_word = []
        mask_idx = []

        if type == "adjacent":
            offset = sen[0].i
            for k in keyword:
                if k.i - offset < len(sen) - 1:
                    mask_cand = sen[k.i - offset + 1]
                    if self._check_mask_candidate(mask_cand, mask_word, keyword, ent_keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)

        elif type == "child":
            for k in keyword:
                mask_candidates = list(k.children)
                # mask_candidates = mask_candidates[:1]
                for mask_cand in mask_candidates:
                    if self._check_mask_candidate(mask_cand, mask_word, keyword, ent_keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)
                        break

        elif type == "child_dep":
            mask_candidates = []
            for k in keyword:
                connected_components = list(k.children)
                dep_in_sentence = [t.dep_ for t in connected_components]
                for d in self.dep_ordering:
                    if d in dep_in_sentence:
                        indices = [idx for idx, dep in enumerate(dep_in_sentence) if dep == d]
                        for idx in indices:
                            mask_candidates.append(connected_components[idx])
                # mask_candidates = mask_candidates
                for mask_cand in mask_candidates:
                    if self._check_mask_candidate(mask_cand, mask_word, keyword, ent_keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)
                        break

        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key= lambda x: x[0])]
        offset = sen[0].i
        mask_idx.sort()
        mask_idx = [m-offset for m in mask_idx]

        return mask_idx, mask_word

    def grammar_component(self, sen, keyword, ent_keyword, ordering_by="dep"):
        max_mask_cnt = len(keyword)
        mask_word = []
        mask_idx = []
        dep_in_sentence = [t.dep_ for t in sen] if ordering_by == "dep" else \
                            [t.pos_ for t in sen]
        mask_candidates = []

        ordering = self.dep_ordering if ordering_by == "dep" else self.pos_ordering
        for d in ordering:
            if d in dep_in_sentence:
                indices = [idx for idx, dep in enumerate(dep_in_sentence) if dep == d]
                for idx in indices:
                    mask_candidates.append(sen[idx])

        if mask_candidates:
            mask_candidates = mask_candidates
            for m_cand in mask_candidates:
                if self._check_mask_candidate(m_cand, mask_word, keyword, ent_keyword):
                    mask_word.append(m_cand)
                    mask_idx.append(m_cand.i)
                    if len(mask_word) == max_mask_cnt:
                        break

        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key=lambda x: x[0])]
        offset = sen[0].i
        mask_idx.sort()
        mask_idx = [m-offset for m in mask_idx]

        return mask_idx, mask_word

    def _check_mask_candidate(self, mask_cand, mask_word, keyword=[], ent_keyword=[],
                                    keyword_ablate=False):
        def _contains_punct(mask_cand):
            return any(p in mask_cand.text for p in punctuation)

        if keyword_ablate:
            if mask_cand not in ent_keyword and not mask_cand.is_punct and not mask_cand.pos_ == "PART" \
                    and not _contains_punct(mask_cand):
                return True
            else:
                return False

        if mask_cand not in keyword and mask_cand not in mask_word \
                and mask_cand not in ent_keyword and not mask_cand.is_punct and not mask_cand.pos_ == "PART"\
                and not _contains_punct(mask_cand) and mask_cand.text not in self.custom_keywords:
            return True
        else:
            return False

    def keyword_disconnected(self, sen, keyword, ent_keyword):
        max_mask_cnt = len(keyword)
        self.num_max_mask.append(max_mask_cnt)
        mask_word = []
        mask_idx = []
        dep_in_sentence = [t.dep_ for t in sen]
        mask_candidates = []

        for d in self.dep_ordering:
            if d in dep_in_sentence:
                indices = [idx for idx, dep in enumerate(dep_in_sentence) if dep == d]
                for idx in indices:
                    mask_candidates.append(sen[idx])

        if mask_candidates:
            mask_candidates = mask_candidates[:max_mask_cnt]
            for m_cand in mask_candidates:
                if self._check_mask_candidate(m_cand, mask_word, keyword, ent_keyword):
                    mask_word.append(m_cand)
                    mask_idx.append(m_cand.i)
                    if len(mask_word) >= max_mask_cnt:
                        break

        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key=lambda x: x[0])]
        offset = sen[0].i
        mask_idx.sort()
        mask_idx = [m-offset for m in mask_idx]

        return mask_idx, mask_word
