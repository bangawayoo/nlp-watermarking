export CUDA_VISIBLE_DEVICES=0
DTYPE="imdb"
NAME="dep-example"
SPACYM="en_core_web_sm"
CKPT=""

KR=0.05
TOPK=4

# Syntatic component
MASK_S="grammar"
MASK_ORDER_BY="dep"
EXCLUDE_CC="F"
K_MASK="na"

mkdir -p "results/ours/${DTYPE}/${NAME}"
cp "$0" "results/ours/${DTYPE}/${NAME}"

METRIC_ONLY="F"
NSAMPLE=5000

EMBED="T"
if [ $EMBED = "T" ] ; then
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
        --keyword_mask $K_MASK -exclude_cc $EXCLUDE_CC -metric_only $METRIC_ONLY
fi

EXTRACT="T"
if [ $EXTRACT = "T" ] ; then
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
        --keyword_mask $K_MASK -exclude_cc $EXCLUDE_CC
fi


# Keyword component
NAME="keyword-example"
MASK_S="keyword_connected"
MASK_ORDER_BY="dep"
K_MASK="adjacent"
EXCLUDE_CC="F"

KR=0.06
TOPK=4

mkdir -p "results/ours/${DTYPE}/${NAME}"
cp "$0" "results/ours/${DTYPE}/${NAME}"

METRIC_ONLY="F"
NSAMPLE=5000

EMBED="T"
if [ $EMBED = "T" ] ; then
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
        --keyword_mask $K_MASK -exclude_cc $EXCLUDE_CC -metric_only $METRIC_ONLY
fi


EXTRACT="T"
if [ $EXTRACT = "T" ] ; then
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
        --keyword_mask $K_MASK -exclude_cc $EXCLUDE_CC
fi