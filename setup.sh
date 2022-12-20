pip install -U pip setuptools wheel
pip install -U 'spacy[cuda-autodetect]'

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf

pip install nltk
python -m nltk.downloader stopwords
python -m nltk.downloader punkt

pip install -r requirements.txt

#conda create -n ta
#conda init bash
#conda activate ta
pip install textattack[tensorflow]

echo "export NLTK_DATA=/workspace/.cache/NLTK_DATA" >> ~/.bashrc
echo "export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers" >> ~/.bashrc
source ~/.bashrc