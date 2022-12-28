export CUDA_VISIBLE_DEVICES=0

DTYPE="dracula"
APCT="0.1"
NUMCORR=10
AUGMENT_TYPE="all"

python ./models/corruption/run_attack.py  --attack_pct $APCT --dtype $DTYPE \
                                          --path2result "./data/${DTYPE}-augmented.txt"\
                                          --attack_type "None" \
                                          -augment True --num_corr_per_sentence $NUMCORR \
                                          --augment_type $AUGMENT_TYPE
