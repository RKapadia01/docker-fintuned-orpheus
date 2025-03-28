FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV MODEL_NAME="rohan2710/bono-orpheus"
ENV TOKENISER_NAME="meta-llama/Llama-3.2-3B-Instruct"
ENV SNAC_MODEL_NAME="hubertsiuzdak/snac_24khz"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy handler script
COPY handler.py /app/handler.py

# Pre-download models at build time (optional, makes container larger but startup faster)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='${SNAC_MODEL_NAME}'); \
    snapshot_download(repo_id='${MODEL_NAME}', \
    allow_patterns=['config.json', '*.safetensors', 'model.safetensors.index.json', \
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'])"

# Expose port for API
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python3", "-m", "handler"]