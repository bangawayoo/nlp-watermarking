DEBUG="False"
EXP_NAME="mask=ours-higher-bpw"
DATA_TYPE="wikitext"
EPOCH=100
MODEL_CKPT=""

KL_TYPE="reverse"
MASKING_TYPE="ours"
MASKING_P=0.15

KR=0.065
MASK_S="grammar"
MASK_ORDER_BY="dep"
K_MASK="adjacent"
EXCLUDE_CC="F"
OPTIM_TOPK="T"

mkdir -p "./ckpt/${DATA_TYPE}/${EXP_NAME}"
cp "$0" "./ckpt/${DATA_TYPE}/${EXP_NAME}"

CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision bf16 \
 train_infill_fast.py --exp_name $EXP_NAME \
                  -debug_mode $DEBUG \
                  --dtype $DATA_TYPE \
                  --num_epochs $EPOCH \
                  --mask_select_method $MASK_S \
                  --mask_order_by $MASK_ORDER_BY \
                  --keyword_mask $K_MASK \
                  --keyword_ratio $KR \
                  --masking_type $MASKING_TYPE \
                  --masking_p $MASKING_P \
                  --kl_type $KL_TYPE \
                  -eval_init True -exclude_cc $EXCLUDE_CC -optimize_topk $OPTIM_TOPK