FROM huggingface/transformers-pytorch-gpu

# Install base utilities
RUN apt-get update && \
    apt-get install -y \
    wget \
    sudo \
    htop \
    make \
    vim \
    tmux \
    openssh-server \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Change ssh config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# set environment variables
ENV NLTK_DATA /workspace/.cache/NLTK_DATA
ENV HF_DATASETS_CACHE /workspace/.cache/huggingface/datasets
ENV TRANSFORMERS_CACHE /workspace/.cache/huggingface/transformers