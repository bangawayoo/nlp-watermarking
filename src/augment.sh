export CUDA_VISIBLE_DEVICES=1

DTYPE="imdb"
APCT="0.1"
python ./models/corruption/run_attack.py  --attack_pct $APCT\
                                          --path2result "./data/${DTYPE}-augmented-full.txt"\
                                          --attack_type "None" \
                                          -augment True --data_type $DTYPE
#--num_sentence 100 \
