DEBUG="False"
EXP_NAME="reverse-kl"
DATA_TYPE="imdb"

accelerate launch train_infill.py --exp_name $EXP_NAME --data_type $DATA_TYPE