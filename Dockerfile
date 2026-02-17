FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# git is required by HuggingFace Spaces build hooks
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Non-root user required by HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Optimise CPU inference for 8 vCPU hardware
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

WORKDIR /home/user/app

# Install pre-built llama-cpp-python CPU wheel (no compilation needed)
RUN pip install --no-cache-dir \
        llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        sse-starlette \
        huggingface_hub \
        httpx

COPY --chown=user:user server.py .
COPY --chown=user:user inference.py .

EXPOSE 7860

CMD ["python", "server.py", "--port", "7860"]
