DEBUG="False"
EVAL_ONLY="False"
MODEL_CKPT=""
EXP_NAME=""

DATA_TYPE="IMDB"


python models/infill.py \
-debug_mode $DEBUG \
-eval_only $EVAL_ONLY \
--model_ckpt $MODEL_CKPT \
--exp_name $EXP_NAME \
--data_type $DATA_TYPE
