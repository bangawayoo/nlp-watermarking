export CUDA_VISIBLE_DEVICES=1
NAME="robust-infill-reverse-kl=115000"
SPACYM="en_core_web_sm"
CKPT="./ckpt/reverse-kl/115000.pth"

python ./ours.py -do_watermark T -embed T --exp_name $NAME --num_sample 100 --spacy_model $SPACYM\
      --model_ckpt $CKPT
#python ./ours.py -do_watermark T -extract T -extract_corrupted F\
#                 --exp_name $NAME --spacy_model $SPACYM

##
SS_THRES=0.98
ATTACKM="insertion deletion substitution"
PCT_RANGE="0.05"

for apct in $PCT_RANGE
do
for attm in $ATTACKM
  do
    CORRUPTION_NAME="watermarked"
    python ./models/corruption/run_attack.py --target_method "ours"\
                                             --attack_pct $apct\
                                             --path2embed "results/ours/imdb/${NAME}/watermarked.txt"\
                                             --attack_type $attm --num_sentence 500 --ss_thres $SS_THRES

    python ./ours.py -do_watermark T -extract T -extract_corrupted T\
                     --corrupted_file_dir "./results/ours/imdb/${NAME}/watermarked-${attm}=${apct}.txt"\
                     --exp_name $NAME --spacy_model $SPACYM --model_ckpt $CKPT
  done
done

