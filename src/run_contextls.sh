export CUDA_VISIBLE_DEVICES=1

NAME="sm_spacy"
SPACYM="en_core_web_sm"

python context-ls.py -embed T --num_sample 100 --exp_name $NAME --spacy_model $SPACYM

#PCT_RANGE="0.05"
#ATTACKM="substitution deletion insertion"
#for attm in $ATTACKM
#do
#  for apct in $PCT_RANGE
#  do
#    ATTACK_PCT=$apct
##    python ./models/corruption/run_attack.py --target_method "context-ls"\
##                                             --attack_pct $ATTACK_PCT\
##                                             --path2embed "results/context-ls/imdb/${NAME}/watermarked.txt"\
##                                             --attack_type $attm --num_sentence 0
#
#    python context-ls.py -extract T -extract_corrupted T --exp_name $NAME\
#                        --corrupted_file_dir "./results/context-ls/imdb/${NAME}/watermarked-${attm}=${ATTACK_PCT}.txt"
#  done
#done
