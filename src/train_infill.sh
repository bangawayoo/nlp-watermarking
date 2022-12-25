export CUDA_VISIBLE_DEVICES=1
DEBUG="False"
EXP_NAME="tmp"
DATA_TYPE="imdb"
EPOCH=200

KR=0.06
MASK_S="keyword_connected"
MASK_ORDER_BY="dep"
K_MASK="child_dep"

mkdir -p "./ckpt/${EXP_NAME}"
cp "$0" "./ckpt/${EXP_NAME}"


#accelerate launch train_infill.py --exp_name $EXP_NAME --data_type $DATA_TYPE --num_epochs $EPOCH
python train_infill.py --exp_name $EXP_NAME --data_type $DATA_TYPE --num_epochs $EPOCH \
      --mask_select_method $MASK_S --mask_order_by $MASK_ORDER_BY --keyword_mask $K_MASK_S \
      --keyword_ratio $KR
