FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y \
    build-essential \
    cifs-utils \
    curl \
    default-jdk \
    dialog \
    dos2unix \
    git \
    sudo

# Install app requirements first to avoid invalidating the cache
COPY requirements.txt setup.py /app/
WORKDIR /app
RUN pip install --user -r requirements.txt && \
    pip install --user entmax && \
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Cache the pretrained BERT model
RUN python -c "from transformers import BertModel; BertModel.from_pretrained('bert-large-uncased-whole-word-masking')"

# Download & cache StanfordNLP
RUN mkdir -p /app/third_party && \
    cd /app/third_party && \
    curl https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip | jar xv

# Now copy the rest of the app
COPY . /app/

# Assume that the datasets will be mounted as a volume into /mnt/data on startup.
# Symlink the data subdirectory to that volume.
ENV CACHE_DIR=/mnt/data
RUN mkdir -p /mnt/data && \
    mkdir -p /app/data && \
    cd /app/data && \
    ln -snf /mnt/data/spider spider && \
    ln -snf /mnt/data/wikisql wikisql

# Convert all shell scripts to Unix line endings, if any
RUN /bin/bash -c 'if compgen -G "/app/**/*.sh" > /dev/null; then dos2unix /app/**/*.sh; fi'

# Extend PYTHONPATH to load WikiSQL dependencies
ENV PYTHONPATH="/app/third_party/wikisql/:${PYTHONPATH}" 

ENTRYPOINT bash
