export CUDA_VISIBLE_DEVICES=0
NAME="dep_ordering-sm"
SPACYM="en_core_web_sm"

#python ./ours.py -do_watermark T -embed T --exp_name $NAME --num_sample 30 --spacy_model $SPACYM
#python ./ours.py -do_watermark T -extract T -extract_corrupted F\
#                 --exp_name $NAME --spacy_model $SPACYM

##
ATTACKM="insertion deletion substitution"
PCT_RANGE="0.05 0.1"
for attm in $ATTACKM
do
for apct in $PCT_RANGE
  do
    CORRUPTION_NAME="watermarked"
    python ./models/corruption/run_attack.py --target_method "ours"\
                                             --attack_pct $apct\
                                             --path2embed "results/ours/imdb/${NAME}/watermarked.txt"\
                                             --attack_type $attm --num_sentence 300

    python ./ours.py -do_watermark T -extract T -extract_corrupted T\
                     --corrupted_file_dir "./results/ours/imdb/${NAME}/watermarked-${attm}=${apct}.txt"\
                     --exp_name $NAME --spacy_model $SPACYM
  done
done

