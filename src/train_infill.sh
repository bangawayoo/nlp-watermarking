
DEBUG="False"
EXP_NAME="mask-ours-soft"
DATA_TYPE="imdb"
EPOCH=100
MODEL_CKPT=""

KL_TYPE="reverse"
MASKING_TYPE="ours"
MASKING_P=0.15

KR=0.05
MASK_S="grammar"
MASK_ORDER_BY="dep"
K_MASK="adjacent"
EXCLUDE_CC="F"
OPTIM_TOPK="F"

mkdir -p "./ckpt/${DATA_TYPE}/${EXP_NAME}"
cp "$0" "./ckpt/${DATA_TYPE}/${EXP_NAME}"

#CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision bf16 \
#  train_infill_ablation.py --exp_name $EXP_NAME \
#                  --dtype $DATA_TYPE \
#                  --num_epochs $EPOCH \
#                  --mask_select_method $MASK_S \
#                  --mask_order_by $MASK_ORDER_BY \
#                  --keyword_mask $K_MASK \
#                  --keyword_ratio $KR \
#                  --masking_type $MASKING_TYPE \
#                  --masking_p $MASKING_P \
#                  --kl_type $KL_TYPE \
#                  -eval_init False

CUDA_VISIBLE_DEVICES="2" accelerate launch --mixed_precision bf16 \
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
                  -eval_init False -exclude_cc $EXCLUDE_CC -optimize_topk $OPTIM_TOPK