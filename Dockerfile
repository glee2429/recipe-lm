FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Non-root user required by HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining serving dependencies
RUN pip install --no-cache-dir \
        transformers \
        peft \
        accelerate \
        fastapi \
        "uvicorn[standard]" \
        sse-starlette \
        huggingface_hub

COPY --chown=user:user server.py .
COPY --chown=user:user inference.py .

EXPOSE 7860

CMD ["python", "server.py", \
     "--adapter", "ClaireLee2429/gemma-2b-recipes-lora", \
     "--port", "7860"]
