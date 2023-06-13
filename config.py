import argparse
import string

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
punctuation = set(string.punctuation)
# riskset = stop.union(punctuation)
riskset = punctuation


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def WatermarkArgs():
    parser = argparse.ArgumentParser(description="For the watermarking module")
    parser.add_argument("-debug_mode", type=str2bool, default=False)
    parser.add_argument("-eval_only", type=str2bool, default=False)
    parser.add_argument("-do_watermark", type=str2bool, default=True)
    parser.add_argument("-train_infill", type=str2bool, default=False)

    parser.add_argument("--model_ckpt", type=str, nargs="?", const="")
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--exp_name", type=str, default="")

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--dtype", type=str, default='imdb')
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--keyword_ratio", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--mask_select_method", type=str, default="",
                        choices=['keyword_disconnected', "keyword_connected", "grammar"])
    parser.add_argument("--mask_order_by", type=str, default="", choices=['dep', 'pos'])
    parser.add_argument("--keyword_mask", type=str, default="adjacent", choices=['adjacent', 'child', 'child_dep', "na"])
    parser.add_argument("-exclude_cc", type=str2bool, default=False)

    return parser


def GenericArgs():
    parser = argparse.ArgumentParser(description="Generic argument parser")
    parser.add_argument("--exp_name", type=str, default="tmp")
    parser.add_argument("-embed", type=str2bool, default=False)
    parser.add_argument("-extract", type=str2bool, default=False)
    parser.add_argument("-extract_corrupted", type=str2bool, default=False)
    parser.add_argument("--corrupted_file_dir", type=str, default="")
    parser.add_argument("--dtype", type=str, default="imdb")
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("-debug_mode", type=str2bool, default=False)
    parser.add_argument("-metric_only", type=str2bool, default=False)

    return parser


def CorruptionArgs():
    parser = argparse.ArgumentParser(description="Generic argument parser")
    parser.add_argument("--target_method", type=str, default="")
    parser.add_argument("-augment", type=str2bool, default=False)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--attack_pct", type=float, default=0.05)
    parser.add_argument("--attack_type", type=str, default="insertion")
    parser.add_argument("--ss_thres", type=float, default=0.98)
    parser.add_argument("--path2embed", type=str, default="imdb")
    parser.add_argument("--path2result", type=str, default="")
    parser.add_argument("--num_sentence", type=int, default=0)
    parser.add_argument("--augment_type", type=str, default="random", choices=['random', 'all'])
    parser.add_argument("--num_corr_per_sentence", type=int, default=1)

    return parser


def InfillArgs():
    parser = argparse.ArgumentParser(description="Training or Evaluating the infill model")
    parser.add_argument("-debug_mode", type=str2bool, default=False)
    parser.add_argument("-eval_only", type=str2bool, default=False)
    parser.add_argument("-eval_init", type=str2bool, default=True)

    parser.add_argument("--model_ckpt", type=str, nargs="?", const="")
    parser.add_argument("--exp_name", type=str, default="")

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--masking_type", type=str, default="ours")
    parser.add_argument("--masking_p", type=float, default=0.15)
    parser.add_argument("--kl_type", type=str, default="forward")
    parser.add_argument("--dtype", type=str, default='imdb')
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("-preprocess_data", type=str2bool, default=False)
    parser.add_argument("-optimize_topk", type=str2bool, default=True)

    return parser