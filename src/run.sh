export CUDA_VISIBLE_DEVICES=0
DTYPE="imdb"
NAME="new/keyword-child-dep"
SPACYM="en_core_web_sm"
CKPT=""

KR=0.06
TOPK=3

MASK_S="keyword_connected"
MASK_ORDER_BY="dep"
K_MASK="child_dep"

mkdir -p "results/ours/${DTYPE}/${NAME}"
cp "$0" "results/ours/${DTYPE}/${NAME}"


NSAMPLE=1000
EMBED="T"
if [ $EMBED = "T" ]
then
  python ./ours.py \
        -do_watermark T \
        -embed T \
        --dtype $DTYPE \
        --exp_name $NAME \
        --num_sample $NSAMPLE \
        --spacy_model $SPACYM \
        --model_ckpt $CKPT \
        --keyword_ratio $KR \
        --topk $TOPK \
        --mask_select_method $MASK_S \
        --mask_order_by $MASK_ORDER_BY \
        --keyword_mask $K_MASK
fi

EXTRACT="F"
if [ $EXTRACT = "T" ]
then
  python ./ours.py -do_watermark T -extract T -extract_corrupted F\
        --exp_name $NAME \
        --dtype $DTYPE \
        --num_sample $NSAMPLE \
        --spacy_model $SPACYM \
        --model_ckpt $CKPT \
        --keyword_ratio $KR \
        --topk $TOPK \
        --mask_select_method $MASK_S \
        --mask_order_by $MASK_ORDER_BY \
        --keyword_mask $K_MASK
fi


CORRUPT="T"
if [ $CORRUPT = "T" ]
then
  SS_THRES=0.98
  ATTACKM="deletion insertion substitution"
  PCT_RANGE="0.025 0.05"
  NUM_SENTENCE=1000

  SEED=$(seq 0 0)
  for seed in $SEED
    do
    for apct in $PCT_RANGE
    do
    for attm in $ATTACKM
      do
        if [ $attm = "deletion" ]
        then
          ncps=1
        else
          ncps=5
        fi
        CORRUPTION_NAME="watermarked"
        python ./models/corruption/run_attack.py --target_method "ours"\
                                                 --attack_pct $apct\
                                                 --path2embed "results/ours/${DTYPE}/${NAME}/watermarked.txt"\
                                                 --attack_type $attm --num_sentence $NUM_SENTENCE --ss_thres $SS_THRES \
                                                 --num_corr_per_sentence $ncps

        python ./ours.py -do_watermark T -extract T -extract_corrupted T --dtype $DTYPE \
                         --corrupted_file_dir "./results/ours/${DTYPE}/${NAME}/watermarked-${attm}=${apct}.txt"\
                         --exp_name $NAME --spacy_model $SPACYM --model_ckpt $CKPT \
                        --keyword_ratio $KR \
                        --topk $TOPK \
                        --mask_select_method $MASK_S \
                        --mask_order_by $MASK_ORDER_BY \
                        --keyword_mask $K_MASK

      done
    done
  done
fi

