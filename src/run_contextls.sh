export CUDA_VISIBLE_DEVICES=1

NAME="sm_spacy"
SPACYM="en_core_web_sm"
DTYPE="wikitext"

mkdir -p "results/context-ls/${DTYPE}/${NAME}"
cp "$0" "results/context-ls/${DTYPE}/${NAME}"

python context-ls.py -embed T --num_sample 100 --exp_name $NAME --spacy_model $SPACYM --dtype $DTYPE

#
#SS_THRES=0.98
#ATTACKM="insertion substitution deletion"
#PCT_RANGE="0.025"
#NUM_SENTENCE=1000
#
#for apct in $PCT_RANGE
#do
#for attm in $ATTACKM
#  do
#    if [ $attm = "deletion" ]
#      then
#        ncps=1
#      else
#        ncps=5
#      fi
#    ATTACK_PCT=$apct
#    python ./models/corruption/run_attack.py --target_method "context-ls"\
#                                             --attack_pct $ATTACK_PCT\
#                                             --path2embed "results/context-ls/${DTYPE}/${NAME}/watermarked.txt"\
#                                             --attack_type $attm --num_sentence $NUM_SENTENCE --ss_thres $SS_THRES \
#                                             --num_corr_per_sentence $ncps
#
#    python context-ls.py -extract T -extract_corrupted T --exp_name $NAME --dtype $DTYPE --num_sample 1000\
#                        --corrupted_file_dir "./results/context-ls/${DTYPE}/${NAME}/watermarked-${attm}=${ATTACK_PCT}.txt"
#  done
#done