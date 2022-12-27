#export CUDA_VISIBLE_DEVICES=1
DEBUG="False"
EXP_NAME="mask=random-forward-p=15"
DATA_TYPE="imdb"
EPOCH=200
MODEL_CKPT=""

KL_TYPE="forward"

MASKING_TYPE="random"
MASKING_P=0.15

KR=0.06
MASK_S="grammar"
MASK_ORDER_BY="dep"
K_MASK="child_dep"

mkdir -p "./ckpt/${EXP_NAME}"
cp "$0" "./ckpt/${EXP_NAME}"

CUDA_VISIBLE_DEVICES="3" accelerate launch --mixed_precision bf16 \
  train_infill.py --exp_name $EXP_NAME \
                  --data_type $DATA_TYPE \
                  --num_epochs $EPOCH \
                  --mask_select_method $MASK_S \
                  --mask_order_by $MASK_ORDER_BY \
                  --keyword_mask $K_MASK \
                  --keyword_ratio $KR \
                  --masking_type $MASKING_TYPE \
                  --masking_p $MASKING_P \
                  --kl_type $KL_TYPE \
                  -eval_init False
