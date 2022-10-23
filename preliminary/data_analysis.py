import pandas as pd
import os
import sys
sys.path.append("/workspace/ilm/")

import pickle

NUM_SAMPLE=30
with open(f"./preliminary/abs_parsed.pkl", 'rb') as f:
    parsed_results = pickle.load(f)[:NUM_SAMPLE] #[idx, tokens, tags, arcs]

with open(f"./preliminary/watermarked_results-{NUM_SAMPLE}.pkl", 'rb') as f:
    watermarked_results = pickle.load(f) #[list[ppls], list[list[words]], list[nums]]

# parsed_results = [parsed_results[0]]
all_tags = []
all_tokens =[]
all_idx = []
for idx, p in enumerate(parsed_results):
    all_idx.extend([idx]*len(p[-2]))
    all_tags.extend(p[-2])
    all_tokens.extend(list(p[1]))

intra_sample_idx = []
all_ppl = []
all_nums = []
all_words = []
for idx, w in enumerate(watermarked_results):
    intra_sample_idx.extend(range(len(w[0])))
    all_ppl.extend(w[0])
    all_words.extend(w[1])
    all_nums.extend(w[2])

data = {k:v for k,v in zip(
    ["corpus_idx", 'intra_idx', "tags", "token", "ppl_delta", "num_of_rep", "replaced_tokens"],
                   [all_idx, intra_sample_idx, all_tags, all_tokens, all_ppl, all_nums, all_words])}
df = pd.DataFrame(data)
df = df.replace("NA", -1)

##
import matplotlib.pyplot as plt


#pick top-k tags
topk = 10
topk_tags = df['tags'].value_counts(sort=True).index.tolist()
topk_tags = [t for t in topk_tags if t != "punct"][:topk]

invalid_cnt = df[df['ppl_delta'] == -1]
invalid_cnt.groupby('tags').count()

filtered_df = df[df['tags'].isin(topk_tags)]
filtered_df = filtered_df[filtered_df['ppl_delta'] != -1]
axes = filtered_df.hist(column=['num_of_rep'], by='tags', bins=5)
fig = axes[0][0].figure
fig.suptitle('Replaced Sentence PPL - Original PPL')
fig.suptitle('# of Replacable Words')
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.9, wspace=0.3, hspace=0.7)
plt.show()

# hist = filtered_df.hist(column=['ppl_delta'], by='tags')
# plt.title('Increase in PPL')
# plt.show()

##
#see examples
# from preliminary.utils import color_text

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

nn = df[df['tags']=="det"].sort_values(by='ppl_delta')
nn_sample = nn.iloc[-3]
c_idx = nn_sample['corpus_idx']
txt = " ".join(parsed_results[c_idx][1])

print(nn_sample['ppl_delta'])
print(color_text(txt, [nn_sample['intra_idx']]))
print(nn_sample['replaced_tokens'])


