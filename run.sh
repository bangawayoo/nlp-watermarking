export CUDA_VISIBLE_DEVICES=0
NAME="mask_both-topk4-trf_spacy_model"
NAME="tmp"
#NAME="mask_ordering_ablation-topk4-sm_spacy_model"
SPACYM="en_core_web_sm"

python ./ours.py -do_watermark T -embed T --exp_name $NAME --num_sample 100 --spacy_model $SPACYM
#
#ATTACK_PCT=0.05
#python ./models/corruption/run_attack.py --target_method "ours"\
#                                         --attack_pct $ATTACK_PCT\
#                                         --path2embed "results/ours/imdb/${NAME}/watermarked.txt"\
#                                         --attack_type "insertion" --num_sentence 0
#
#python ./ours.py -do_watermark T -extract T -extract_corrupted T\
#                 --corrupted_file_dir "./results/ours/imdb/${NAME}/watermarked-corrupted=${ATTACK_PCT}.txt"\
#                 --exp_name $NAME --spacy_model $SPACYM
#
#
#ATTACK_PCT=0.1
#python ./models/corruption/run_attack.py --target_method "ours"\
#                                         --attack_pct $ATTACK_PCT\
#                                         --path2embed "results/ours/imdb/${NAME}/watermarked.txt"\
#                                         --attack_type "insertion" --num_sentence 0
#
#python ./ours.py -do_watermark T -extract T -extract_corrupted T\
#                 --corrupted_file_dir "./results/ours/imdb/${NAME}/watermarked-corrupted=${ATTACK_PCT}.txt"\
#                 --exp_name $NAME --spacy_model $SPACYM