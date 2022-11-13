DEBUG="True"
EVAL_ONLY="True"
MODEL_CKPT="bert-base-cased"
EXP_NAME="1000-samples"
DATA_TYPE="imdb"


python models/infill.py \
-debug_mode $DEBUG \
-eval_only $EVAL_ONLY \
--model_ckpt $MODEL_CKPT \
--exp_name $EXP_NAME \
--data_type $DATA_TYPE


