export CUDA_VISIBLE_DEVICES=1
DTYPE="imdb"
NAME="exp"
SPACYM="en_core_web_sm"
CKPT=""

KR=0.05
TOPK=4

MASK_S="grammar"
MASK_ORDER_BY="dep"
K_MASK="adjacent"
EXCLUDE_CC="T"

mkdir -p "results/ours/${DTYPE}/${NAME}"
cp "$0" "results/ours/${DTYPE}/${NAME}"

METRIC_ONLY="F"
NSAMPLE=100

EMBED="T"
if [ $EMBED = "T" ] ; then
  python ./demo.py \
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
