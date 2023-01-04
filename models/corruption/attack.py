from datasets import load_dataset

from module import Attacker
from config import CorruptionArgs, WatermarkArgs
from utils.dataset_utils import preprocess2sentence, preprocess_txt, get_dataset

parser = CorruptionArgs()
args, _ = parser.parse_known_args()

# load data
attack_type = args.attack_type
attack_percentage = args.attack_pct
method = args.target_method
path2embed = args.path2embed

if args.augment:
    assert len(args.path2result) > 0, "Save path is empty"
    path2result = args.path2result
    constraint_kwargs = {'use': args.ss_thres, 'num_sentence': args.num_sentence}
    attacker = Attacker(attack_type, attack_percentage, path2embed, path2result, constraint_kwargs,
                        augment_data_flag=True, num_corr_per_sentence=args.num_corr_per_sentence, args=args)

    wm_parser = WatermarkArgs()
    wm_args, _ = wm_parser.parse_known_args()

    dtype = wm_args.dtype
    corpus, _, numsample2use = get_dataset(dtype)
    cover_texts = preprocess_txt(corpus)
    cover_texts = preprocess2sentence(cover_texts, dtype + "-train", 0, numsample2use['train'],
                                      spacy_model=wm_args.spacy_model)
    attacker.augment_data(cover_texts)

elif method == "awt":
    import spacy
    path2embed = "./data/awt/stego-novel-0.05.txt"
    with open(path2embed, "r") as f:
        text = f.readlines()[0]

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text_with_ws for sent in doc.sents]

    formatted_sentences = []
    for sen_idx, sent in enumerate(sentences):
        formatted_sentences.append((sen_idx, 0, 0, 0 , sent, 0, 0))

    constraint_kwargs = {'use': args.ss_thres, 'num_sentence': args.num_sentence}
    path2result = f"./data/awt/awt-novels-corrupted-{attack_type}.txt"
    attacker = Attacker(attack_type, attack_percentage, path2embed, path2result, constraint_kwargs,
                        num_corr_per_sentence=args.num_corr_per_sentence, args=args)
    attacker.attack_sentence(texts=formatted_sentences)

    # reformat to the original file
    corrupted_text_list = []
    with open(path2result, "r") as reader:
        for line in reader:
            corrupted_text_list.append(line)

    corrupted_text = "".join([sen.rstrip("\n") for sen in corrupted_text_list])
    path2result = f"./data/awt/postprocessed-awt-novels-corrupted-{attack_type}.txt"
    with open(path2result, "w") as f:
        f.write(corrupted_text)


else:
    if args.exp_name:
        path2result = f"./{path2embed.split('.')[0]}-{args.exp_name}-{attack_type}={attack_percentage}.txt"
    else:
        path2result = f"./{path2embed.split('.')[0]}-{attack_type}={attack_percentage}.txt"


    constraint_kwargs = {'use': args.ss_thres, 'num_sentence': args.num_sentence}
    attacker = Attacker(attack_type, attack_percentage, path2embed, path2result, constraint_kwargs,
                        num_corr_per_sentence=args.num_corr_per_sentence)
    attacker.attack_sentence()
