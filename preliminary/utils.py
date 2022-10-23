try:
    import yake
except:
    print("Error importing yake")
import re
import torch
# from ilm.infer import sample_from_logits
import copy
import torch.nn.functional as F

#https://www.kaggle.com/code/rowhitswami/keywords-extraction-using-tf-idf-method
def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def get_tfidf_keywords(vectorizer, feature_names, doc, topk=10):
    """Return top k keywords from a doc using TF-IDF method"""

    # generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo()) # dict of {vocab idx: score}
    result = {}
    splited_doc = doc.split(" ")
    for idx, _ in sorted_items[:topk]:
        result[splited_doc.index(feature_names[idx])] = feature_names[idx]

    return result

def get_YAKE_keywords(doc, topk=10, **kwargs):
    kw_extractor = yake.KeywordExtractor(**kwargs)
    keywords = kw_extractor.extract_keywords(doc) # dict of {phrase: score}

    result = {}
    splited_doc = doc.split(" ")

    for k in keywords[:topk]:
        word = k[0]
        if len(word.split(" ")) > 1:
            word = word.split(" ")[0]
        result[splited_doc.index(word)] = k[0]

    return result

def clean_text(text):
    PUNCTUATION = """!"#$%&'()*+,./:;<=>?@[\]^_`{|}~"""
    """Doc cleaning"""
    text = text.strip()
    # Lowering text
    text = text.lower()

    # Removing punctuation
    text = "".join([c for c in text if c not in PUNCTUATION])

    # Removing whitespace and newlines
    text = re.sub('\s+', ' ', text)

    return text

# def infill_with_ilm(
#         model,
#         special_tokens_to_ids,
#         x,
#         num_infills=1,
#         max_sequence_length=256,
#         nucleus=0.95, deterministic=False):
#     _sep_id = special_tokens_to_ids['<|startofinfill|>']
#     _end_span_id = special_tokens_to_ids['<|endofinfill|>']
#     _special_ids = special_tokens_to_ids.values()
#
#     # Make sure example doesn't already ends with [sep]
#     if x[-1] == _sep_id:
#         x = x[:-1]
#
#     # Count number of blanks
#     blank_idxs = []
#     for i, tok_id in enumerate(x):
#         if tok_id in _special_ids:
#             blank_idxs.append(i)
#     k = len(blank_idxs)
#     if k == 0:
#         raise ValueError()
#
#     # Decode until we have that many blanks
#     with torch.no_grad():
#         device = next(model.parameters()).device
#         context = torch.tensor(x + [_sep_id], dtype=torch.long, device=device).unsqueeze(0).repeat(num_infills, 1)
#
#         terminated = []
#
#         while context.shape[0] > 0:
#             logits = model(context)[0][:, -1]
#             if deterministic:
#                 next_tokens = select_from_logits(logits, nucleus=None, topk=5)
#                 next_tokens = torch.cat((next_tokens, torch.full_like(next_tokens, _end_span_id, device=next_tokens)))
#
#             else:
#                 next_tokens = sample_from_logits(logits, nucleus=nucleus)
#             context = torch.cat((context, next_tokens), dim=1)
#
#             num_predicted_spans = (context == _end_span_id).long().sum(dim=1)
#
#             terminate_expected = num_predicted_spans >= k
#             terminate_toolong = torch.ones_like(context).long().sum(dim=1) >= max_sequence_length
#             terminate = terminate_expected | terminate_toolong
#
#             if torch.any(terminate):
#                 terminated_seqs = context[terminate, len(x) + 1:]
#                 terminated.extend([list(s) for s in terminated_seqs.cpu().numpy()])
#                 context = context[~terminate, :]
#
#     # Collect generated spans
#     generated_spans = []
#     for gen in terminated:
#         spans = []
#         while _end_span_id in gen:
#             spans.append(gen[:gen.index(_end_span_id)])
#             gen = gen[gen.index(_end_span_id) + 1:]
#         while len(spans) < k:
#             spans.append([])
#         generated_spans.append(spans)
#
#     # Insert into context
#     generated = []
#     generated_indices = []
#     for spans in generated_spans:
#         context = copy.deepcopy(x)
#         indices = []
#         offset = 0
#         for i, j in enumerate(blank_idxs[::-1]):
#             del context[j]
#             context[j:j] = spans[k - 1 - i]
#         for i, j in enumerate(blank_idxs):
#             indices.extend(range(j + offset, j + len(spans[i]) + offset))
#             offset += len(spans[i]) - 1
#         generated.append(context)
#         generated_indices.append(indices)
#
#     return generated, generated_indices, terminated, generated_spans


def select_from_logits(
        logits,
        temp=1.,
        topk=None,
        nucleus=1.):
    if temp == 0:
        return torch.argmax(logits, dim=-1).unsqueeze(-1)
    elif temp != 1:
        logits /= temp

    probs = F.softmax(logits, dim=-1)

    if topk is not None:
        top_probs = torch.topk(probs, topk)
        mask = F.one_hot(top_probs.indices, probs.shape[-1]).float()
        mask = mask.sum(dim=1)
        probs *= mask
        probs /= probs.sum(dim=-1)

    if nucleus != 1:
        probs_sorted = torch.sort(probs, descending=True, dim=-1)
        sorted_indices = probs_sorted.indices
        sorted_values = probs_sorted.values

        cumsum = torch.cumsum(sorted_values, dim=-1)
        ks = (cumsum < nucleus).long().sum(dim=-1)
        ks = torch.max(ks, torch.ones_like(ks))

        # TODO: Make this more efficient using gather
        ks = F.one_hot(ks, probs.shape[-1]).float()
        cutoffs = (sorted_values * ks).sum(-1)

        mask = (probs > cutoffs.unsqueeze(1)).float()
        probs *= mask

        probs /= probs.sum(keepdim=True, dim=-1)

    print(probs)
    print(probs.shape)
    next_tokens = torch.nonzero(probs)[:,1].view(-1, 1)

    return next_tokens


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def color_text(text, indices=[0], colorcode="OKBLUE"):
    if not isinstance(text, str):
        text = str(text)
    splited_text = text.split(' ')
    colorcode = eval(f"bcolors.{colorcode}")
    for idx in indices:
        splited_text[idx] = f"{colorcode}{splited_text[idx]}{bcolors.ENDC}"
    return " ".join(splited_text)


def find_diff_word(text, ref_text):
    """
    Currently only works when a single word is inserted
    """
    idx = 0
    for word, ref_word in zip(text.split(" "), ref_text.split(" ")):
        if word != ref_word:
            return idx, word
        idx += 1

    return None, None

def flatten_list(nested_list):
    output = []
    for l in nested_list:
        output.extend(l)