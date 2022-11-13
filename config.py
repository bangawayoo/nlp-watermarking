import argparse
import string

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
punctuation = set(string.punctuation)
riskset = stop.union(punctuation)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def InfillArgs():
    parser = argparse.ArgumentParser(description="Training or Evaluating the infill model")
    parser.add_argument("-debug_mode", type=str2bool, default=False)
    parser.add_argument("-eval_only", type=str2bool, default=False)
    parser.add_argument("-do_watermark", type=str2bool, default=False)

    parser.add_argument("--model_ckpt", type=str, nargs="?", const="bert-base-cased")
    parser.add_argument("--exp_name", type=str, default="")

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--data_type", type=str, default='imdb')
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")

    return parser

def GenericArgs():
    parser = argparse.ArgumentParser(description="Generic argument parser")
    parser.add_argument("--exp_name", type=str, default="")

    return parser
