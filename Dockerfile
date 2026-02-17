FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Build tools for C extensions (uvloop, httptools, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Non-root user required by HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Optimise CPU inference for 8 vCPU hardware
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

WORKDIR /home/user/app

# Install CPU-only PyTorch first, then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
        transformers \
        peft \
        accelerate \
        fastapi \
        "uvicorn[standard]" \
        sse-starlette \
        huggingface_hub \
        httpx

COPY --chown=user:user server.py .
COPY --chown=user:user inference.py .

EXPOSE 7860

CMD ["python", "server.py", \
     "--adapter", "ClaireLee2429/gemma-2b-recipes-lora", \
     "--port", "7860"]
