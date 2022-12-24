export CUDA_VISIBLE_DEVICES=3
DEBUG="False"
EXP_NAME="reverse-kl-logit-loss-longer"
DATA_TYPE="imdb"
EPOCH=500

mkdir -p "./ckpt/${EXP_NAME}"
cp "$0" "./ckpt/${EXP_NAME}"


#accelerate launch train_infill.py --exp_name $EXP_NAME --data_type $DATA_TYPE --num_epochs $EPOCH
python train_infill.py --exp_name $EXP_NAME --data_type $DATA_TYPE --num_epochs $EPOCH