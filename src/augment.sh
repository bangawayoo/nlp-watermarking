export CUDA_VISIBLE_DEVICES=1

DTYPE="imdb"
APCT="0.1"
python ./models/corruption/run_attack.py  --attack_pct $APCT\
                                          --path2result "./data/${DTYPE}-augmented.txt"\
                                          --num_sentence 50000 \
                                          --attack_type "None" \
                                          -augment True --data_type $DTYPE