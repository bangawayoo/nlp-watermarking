export CUDA_VISIBLE_DEVICES=0

DTYPE="wikitext"
APCT="0.1"
NUMCORR=5
AUGMENT_TYPE="random"

python ./models/corruption/attack.py  --attack_pct $APCT --dtype $DTYPE \
                                          --path2result "./data/${DTYPE}-augmented.txt"\
                                          --attack_type "None" \
                                          -augment True --num_corr_per_sentence $NUMCORR \
                                          --augment_type $AUGMENT_TYPE --num_sentence 100000


