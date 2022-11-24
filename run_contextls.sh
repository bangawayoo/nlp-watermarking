export CUDA_VISIBLE_DEVICES=1

python context-ls.py -embed T --num_sample 100

ATTACK_PCT=0.05
python ./models/corruption/run_attack.py --target_method "context-ls"\
                                         --attack_pct $ATTACK_PCT\
                                         --path2embed "results/context-ls/imdb/${NAME}/watermarked.txt"\
                                         --attack_type "insertion" --num_sentence 0

python context-ls.py -extract T -extract_corrupted T\
                    --corrupted_file_dir "./results/context-ls/imdb/${NAME}/watermarked-corrupted=${ATTACK_PCT}.txt"


ATTACK_PCT=0.1
python ./models/corruption/run_attack.py --target_method "context-ls"\
                                         --attack_pct $ATTACK_PCT\
                                         --path2embed "results/context-ls/imdb/${NAME}/watermarked.txt"\
                                         --attack_type "insertion" --num_sentence 0
python context-ls.py -extract T -extract_corrupted T\
                    --corrupted_file_dir "./results/context-ls/imdb/${NAME}/watermarked-corrupted=${ATTACK_PCT}.txt"

