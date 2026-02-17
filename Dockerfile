FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Build tools for llama-cpp-python compilation + git for HF Spaces
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

# Non-root user required by HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Optimise CPU inference for 8 vCPU hardware
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

WORKDIR /home/user/app

# Install llama-cpp-python (compiles from source, ~10 min)
RUN pip install --no-cache-dir llama-cpp-python
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
