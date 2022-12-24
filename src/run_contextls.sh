export CUDA_VISIBLE_DEVICES=0

NAME="tmp"
SPACYM="en_core_web_sm"

mkdir -p "results/context-ls/imdb/${NAME}"
cp "$0" "results/context-ls/imdb/${NAME}"

python context-ls.py -embed T --num_sample 1000 --exp_name $NAME --spacy_model $SPACYM
exit


SS_THRES=0.98
ATTACKM="insertion deletion substitution"
PCT_RANGE="0.05"

for apct in $PCT_RANGE
do
for attm in $ATTACKM
  do
    ATTACK_PCT=$apct
    python ./models/corruption/run_attack.py --target_method "context-ls"\
                                             --attack_pct $ATTACK_PCT\
                                             --path2embed "results/context-ls/imdb/${NAME}/watermarked.txt"\
                                             --attack_type $attm --num_sentence 1000 --ss_thres $SS_THRES


    python context-ls.py -extract T -extract_corrupted T --exp_name $NAME\
                        --corrupted_file_dir "./results/context-ls/imdb/${NAME}/watermarked-${attm}=${ATTACK_PCT}.txt"
  done
done
