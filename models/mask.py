import random


class MaskSelector:
    def __init__(self, **kwargs):
        self.kwargs = {}
        for k, v in kwargs.items():
            self.kwargs[k] = v

        self.dep_ordering = ["expl", 'auxpass', 'cc', 'agent', 'mark' , 'prep', 'aux', 'det', 'prt', 'acl']

    def return_mask(self, sen, keyword, ent_keyword):
        """
        keyword: List[Spacy tokens]
        """
        # mask is selected dependent on the keywords
        mask_word = []
        mask_idx = []

        for k in keyword:
            mask_candidates = list(k.children) + [k.head]
            mask_ordering = []
            for m in mask_candidates:
                if m.dep_ in self.dep_ordering:
                    mask_ordering.append(self.dep_ordering.index(m.dep_))
                else:
                    mask_ordering.append(len(self.dep_ordering))


            mask_candidates = [x for _, x in sorted(zip(mask_ordering, mask_candidates), key=lambda pair: pair[0])]
            # limit the number of candidates to match target bpw
            mask_candidates = mask_candidates[:1]

            if mask_candidates:
                for m_cand in mask_candidates:
                    if m_cand not in keyword and m_cand not in mask_word\
                    and m_cand not in ent_keyword and not m_cand.is_punct and not m_cand.pos_ == "PART":
                        mask_word.append(m_cand)
                        mask_idx.append(m_cand.i)

        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key= lambda x: x[0])]
        offset = sen[0].i
        mask_idx.sort()
        mask_idx = [m-offset for m in mask_idx]

        return mask_idx, mask_word