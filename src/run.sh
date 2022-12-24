export CUDA_VISIBLE_DEVICES=0
DTYPE="wikitext"
NAME="test-embed"
SPACYM="en_core_web_sm"
CKPT=""

KR=0.07
TOPK=5

MASK_S="grammar"
MASK_ORDER_BY="dep"
K_MASK="adjacent"


mkdir -p "results/ours/${DTYPE}/${NAME}"
cp "$0" "results/ours/${DTYPE}/${NAME}"

python ./ours.py \
      -do_watermark T \
      -embed T \
      --dtype $DTYPE \
      --exp_name $NAME \
      --num_sample 1000 \
      --spacy_model $SPACYM \
      --model_ckpt $CKPT \
      --keyword_ratio $KR \
      --topk $TOPK \
      --mask_select_method $MASK_S \
      --mask_order_by $MASK_ORDER_BY \
      --keyword_mask $K_MASK



#python ./ours.py -do_watermark T -extract T -extract_corrupted F\
#      --exp_name $NAME \
#      --num_sample 100 \
#      --spacy_model $SPACYM \
#      --model_ckpt $CKPT \
#      --keyword_ratio $KR \
#      --topk $TOPK \
#      --mask_select_method $MASK_S \
#      --mask_order_by $MASK_ORDER_BY \
#      --keyword_mask $K_MASK


#SS_THRES=0.98
#ATTACKM="insertion deletion substitution"
#PCT_RANGE="0.05"
#NUM_SENTENCE=1000
#
#SEED=$(seq 0 2)
#for seed in $SEED
#  do
#  for apct in $PCT_RANGE
#  do
#  for attm in $ATTACKM
#    do
#      CORRUPTION_NAME="watermarked"
#      python ./models/corruption/run_attack.py --target_method "ours"\
#                                               --attack_pct $apct\
#                                               --path2embed "results/ours/${DTYPE}/${NAME}/watermarked.txt"\
#                                               --attack_type $attm --num_sentence $NUM_SENTENCE --ss_thres $SS_THRES \
#                                               --num_corr_per_sentence 5
#
#      python ./ours.py -do_watermark T -extract T -extract_corrupted T\
#                       --corrupted_file_dir "./results/ours/${DTYPE}/${NAME}/watermarked-${attm}=${apct}.txt"\
#                       --exp_name $NAME --spacy_model $SPACYM --model_ckpt $CKPT \
#                      --keyword_ratio $KR \
#                      --topk $TOPK \
#                      --mask_select_method $MASK_S \
#                      --mask_order_by $MASK_ORDER_BY \
#                      --keyword_mask $K_MASK
#    done
#  done
#done
#
